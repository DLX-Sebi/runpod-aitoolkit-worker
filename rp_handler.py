"""RunPod Serverless worker: AI Toolkit (ostris) LoRA training.

Thin wrapper over trainer_core.py (shared with pod_train.py).
Tasks: train (default) / seed / ls — see README.
"""

import os
import subprocess

VOLUME = "/runpod-volume"
os.environ.setdefault("HF_HOME", os.path.join(VOLUME, "hf"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import runpod  # noqa: E402

import trainer_core as core  # noqa: E402


def handler(job):
    inp = job.get("input") or {}
    task = inp.get("task", "train")
    if task == "seed":
        return core.seed_repos(inp.get("repos", []))
    if task == "ls":
        path = inp.get("path", VOLUME)
        r = subprocess.run(["du", "-h", "-d", "3", path], capture_output=True, text=True)
        return {"du": r.stdout[-8000:], "err": r.stderr[-1000:]}

    def progress(msg):
        try:
            runpod.serverless.progress_update(job, msg)
        except Exception:
            pass

    return core.run_train(inp, job.get("id", "local"), progress_cb=progress)


runpod.serverless.start({"handler": handler})
