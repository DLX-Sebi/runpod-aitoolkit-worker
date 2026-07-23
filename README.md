# runpod-aitoolkit-worker

RunPod **Serverless** worker for [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) LoRA training (DLX AI Studio — Identity Training with full control: checkpoint picking, rank/LR/resolution).

## How it runs

No custom image. The endpoint template uses the public `ostris/aitoolkit` image with a start command that installs the runpod SDK and pulls `rp_handler.py` from this repo at boot:

```bash
bash -c "pip install --no-cache-dir --break-system-packages runpod boto3 && \
  curl -sL https://raw.githubusercontent.com/DLX-Sebi/runpod-aitoolkit-worker/main/rp_handler.py -o /rp_handler.py && \
  python -u /rp_handler.py"
```

Base models live on a **network volume** (mounted at `/runpod-volume`, `HF_HOME=/runpod-volume/hf`) so nothing is re-downloaded per job. Handler updates = git push, no rebuild.

A `Dockerfile` is included for the future baked-image variant (needs a registry with `write:packages`).

## Tasks

| task | input | result |
|------|-------|--------|
| `train` (default) | `name, base (z-image-turbo\|z-image\|z-image-deturbo\|flux-dev), dataset_url (zip), trigger_word, steps, rank, lr, resolution, save_every, batch_size, sample_prompts?, config_overrides?` | every checkpoint `.safetensors` uploaded to S3 (MojoIce) as it appears + presigned URLs returned |
| `seed` | `repos: [hf repo ids]` | downloads bases into the volume HF cache (run once) |
| `ls` | `path?` | `du` of the volume (debug) |

## Endpoint env

`MOJOICE_ACCESS_KEY`, `MOJOICE_SECRET_KEY`, `MOJOICE_ENDPOINT`, `S3_BUCKET`, `S3_PREFIX`, `HF_TOKEN` (for gated repos).
