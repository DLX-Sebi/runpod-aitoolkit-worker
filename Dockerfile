# RunPod Serverless worker: AI Toolkit LoRA training (DLX)
# Base = ostris's official image (torch 2.9.1 cu128, arch list includes sm_120/5090,
# ai-toolkit at /app/ai-toolkit) — we only add the runpod handler layer on top.
FROM ostris/aitoolkit:latest

RUN pip install --no-cache-dir --break-system-packages runpod boto3

# bases live on the network volume, mounted at /runpod-volume on serverless
ENV HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
