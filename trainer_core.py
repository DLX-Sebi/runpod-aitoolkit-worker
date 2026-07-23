"""Shared AI Toolkit training core — used by rp_handler.py (serverless) and
pod_train.py (ephemeral on-demand pod). No runpod SDK dependency here.

HF_HOME must come from the environment: /runpod-volume/hf on serverless,
/workspace/hf on pods (both point at the same network volume).
"""

import os
import shutil
import subprocess
import threading
import time
import traceback
import zipfile

# boto3 >=1.36 defaults to chunked uploads + flexible checksums that
# S3-compatibles (MojoIce) reject with MissingContentLength — force legacy mode
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

import boto3
from boto3.s3.transfer import TransferConfig
import requests
import yaml

AI_TOOLKIT = "/app/ai-toolkit"
S3_BUCKET = os.environ.get("S3_BUCKET", "dlxai-gallery")
S3_PREFIX = os.environ.get("S3_PREFIX", "aitoolkit")

ARCH_MAP = {
    # base -> (arch, name_or_path, extra model fields)
    "z-image-turbo": ("zimage:turbo", "Tongyi-MAI/Z-Image-Turbo", {
        "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
    }),
    "z-image": ("zimage", "Tongyi-MAI/Z-Image", {}),
    "z-image-deturbo": ("zimage:deturbo", "ostris/Z-Image-De-Turbo", {
        "extras_name_or_path": "Tongyi-MAI/Z-Image-Turbo",
    }),
    "flux-dev": ("flux", "black-forest-labs/FLUX.1-dev", {}),
}

# single-part PUT up to 1GB — multipart UploadPart is what MojoIce chokes on
XFER = TransferConfig(multipart_threshold=1024 ** 3)


def s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MOJOICE_ENDPOINT"],
        aws_access_key_id=os.environ["MOJOICE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MOJOICE_SECRET_KEY"],
    )


def upload_and_sign(local_path, key):
    c = s3()
    c.upload_file(local_path, S3_BUCKET, key, Config=XFER)
    return c.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=7 * 86400
    )


def build_config(inp, dataset_dir, output_dir):
    base = inp.get("base", "z-image-turbo")
    if base not in ARCH_MAP:
        raise ValueError(f"unknown base '{base}', valid: {list(ARCH_MAP)}")
    arch, name_or_path, model_extra = ARCH_MAP[base]

    name = inp["name"]
    steps = int(inp.get("steps", 3000))
    save_every = int(inp.get("save_every", 250))
    rank = int(inp.get("rank", 32))
    resolution = inp.get("resolution", [768, 1024])
    if isinstance(resolution, int):
        resolution = [resolution]

    sample_prompts = inp.get("sample_prompts") or []
    sample_cfg = {
        "sampler": "flowmatch",
        "sample_every": int(inp.get("sample_every", save_every)),
        "width": 1024, "height": 1024,
        "samples": [{"prompt": p} for p in sample_prompts],
        "guidance_scale": 1 if arch == "zimage:turbo" else 4,
        "sample_steps": 9 if arch == "zimage:turbo" else 25,
        "walk_seed": True, "seed": 42,
    }

    model = {
        "arch": arch,
        "name_or_path": name_or_path,
        "quantize": True, "qtype": "qfloat8",
        "quantize_te": True, "qtype_te": "qfloat8",
        "low_vram": bool(inp.get("low_vram", False)),
        "model_kwargs": {},
    }
    model.update(model_extra)

    cfg = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "diffusion_trainer",
                "training_folder": output_dir,
                "model": model,
                "device": "cuda",
                "trigger_word": inp.get("trigger_word") or None,
                "performance_log_every": 50,
                "network": {
                    "type": "lora",
                    "linear": rank, "linear_alpha": rank,
                },
                "save": {
                    "dtype": "bf16",
                    "save_every": save_every,
                    # keep every checkpoint on disk — that's the whole point
                    "max_step_saves_to_keep": steps // save_every + 2,
                    "push_to_hub": False,
                },
                "datasets": [{
                    "folder_path": dataset_dir,
                    "caption_ext": "txt",
                    "caption_dropout_rate": float(inp.get("caption_dropout_rate", 0.05)),
                    "cache_latents_to_disk": True,
                    "resolution": resolution,
                    "num_repeats": 1,
                }],
                "train": {
                    "batch_size": int(inp.get("batch_size", 1)),
                    "steps": steps,
                    "gradient_accumulation": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "timestep_type": "weighted" if arch.startswith("zimage") else "sigmoid",
                    "optimizer": "adamw8bit",
                    "optimizer_params": {"weight_decay": 1e-4},
                    "lr": float(inp.get("lr", 1e-4)),
                    "dtype": "bf16",
                    "disable_sampling": not sample_prompts,
                    "ema_config": {"use_ema": False, "ema_decay": 0.99},
                },
                "sample": sample_cfg,
            }],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }

    # arbitrary deep overrides straight from the request (full AI Toolkit control)
    def deep_merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_merge(dst[k], v)
            else:
                dst[k] = v
    deep_merge(cfg, inp.get("config_overrides") or {})
    return cfg


def watch_and_upload(output_dir, name, state, stop_evt, progress_cb):
    """Upload checkpoints AND sample images as soon as their size is stable."""
    sizes = {}
    while not stop_evt.is_set():
        stop_evt.wait(20)
        try:
            for root, _dirs, files in os.walk(output_dir):
                for f in files:
                    is_ckpt = f.endswith(".safetensors")
                    is_sample = "/samples" in root.replace("\\", "/") and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    if not (is_ckpt or is_sample):
                        continue
                    bucket_map = state["uploaded"] if is_ckpt else state["samples"]
                    if f in bucket_map:
                        continue
                    p = os.path.join(root, f)
                    sz = os.path.getsize(p)
                    if sizes.get(f) == sz and sz > 0:  # stable since last pass
                        sub = "" if is_ckpt else "samples/"
                        key = f"{S3_PREFIX}/{name}/{sub}{f}"
                        url = upload_and_sign(p, key)
                        bucket_map[f] = {"key": key, "url": url, "bytes": sz}
                        try:
                            progress_cb(f"uploaded {'sample' if is_sample else 'ckpt'} {f}")
                        except Exception:
                            pass
                    sizes[f] = sz
        except Exception:
            traceback.print_exc()


def run_train(inp, job_id, progress_cb=print):
    """Full training run. Returns the result dict; raises on hard failure."""
    work = f"/tmp/job_{job_id}"
    dataset_dir = os.path.join(work, "dataset")
    output_dir = os.path.join(work, "output")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. dataset
    zpath = os.path.join(work, "dataset.zip")
    with requests.get(inp["dataset_url"], stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(zpath, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dataset_dir)
    entries = os.listdir(dataset_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(dataset_dir, entries[0])):
        dataset_dir = os.path.join(dataset_dir, entries[0])
    n_imgs = len([f for f in os.listdir(dataset_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
    if n_imgs == 0:
        raise ValueError("dataset zip contains no images")

    # 2. config
    cfg = build_config(inp, dataset_dir, output_dir)
    cfg_path = os.path.join(work, "config.yaml")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    # 3. live uploader
    state = {"uploaded": {}, "samples": {}}
    stop_evt = threading.Event()
    t = threading.Thread(target=watch_and_upload,
                         args=(output_dir, cfg["config"]["name"], state, stop_evt, progress_cb),
                         daemon=True)
    t.start()

    # 4. train
    t0 = time.time()
    proc = subprocess.run(
        ["python", "run.py", cfg_path],
        cwd=AI_TOOLKIT, env=dict(os.environ), capture_output=True, text=True)
    dur = round(time.time() - t0, 1)

    # 5. final sweep
    stop_evt.set()
    t.join(timeout=30)
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".safetensors") and f not in state["uploaded"]:
                p = os.path.join(root, f)
                key = f"{S3_PREFIX}/{cfg['config']['name']}/{f}"
                url = upload_and_sign(p, key)
                state["uploaded"][f] = {"key": key, "url": url,
                                        "bytes": os.path.getsize(p)}

    tail = "\n".join((proc.stdout or "").splitlines()[-40:])
    err_tail = "\n".join((proc.stderr or "").splitlines()[-40:])
    shutil.rmtree(work, ignore_errors=True)

    if proc.returncode != 0 and not state["uploaded"]:
        raise RuntimeError(f"training failed rc={proc.returncode}\nSTDOUT:\n{tail}\nSTDERR:\n{err_tail}")

    return {
        "name": cfg["config"]["name"],
        "images": n_imgs,
        "seconds": dur,
        "returncode": proc.returncode,
        "checkpoints": state["uploaded"],
        "samples": state["samples"],
        "log_tail": tail if proc.returncode != 0 else tail[-2000:],
    }


def seed_repos(repos):
    from huggingface_hub import snapshot_download
    hf_home = os.environ.get("HF_HOME", "/workspace/hf")
    os.makedirs(hf_home, exist_ok=True)
    out = {}
    for repo in repos:
        t0 = time.time()
        path = snapshot_download(repo, token=os.environ.get("HF_TOKEN") or None)
        out[repo] = {"path": path, "seconds": round(time.time() - t0, 1)}
    total = subprocess.run(["du", "-sh", hf_home], capture_output=True, text=True)
    out["volume_hf_size"] = total.stdout.strip()
    return out
