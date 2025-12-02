#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
rm -rf /stable-diffusion-webui/models && \
  ln -s /runpod-volume/stable-diffusion-webui/models /stable-diffusion-webui/models
  
echo "Starting WebUI API"
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true
python /stable-diffusion-webui/webui.py \
  --xformers \
  --no-half-vae \
  --skip-python-version-check \
  --skip-torch-cuda-test \
  --skip-install \
  --api-log \
  --opt-sdp-attention \
  --disable-safe-unpickle \
  --port 3000 \
  --api \
  --nowebui \
  --skip-version-check \
  --no-hashing \
  --no-download-sd-model &

echo "Starting RunPod Handler"
python -u /handler.py
