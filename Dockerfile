# ComfyUI + FLUX.1-dev for RunPod, targeting RTX 5090 (Blackwell, sm_120).
#
# Blackwell needs CUDA 12.8 kernels. Torch built for cu121/cu124 will load
# fine and then die at the first matmul with
#   "no kernel image is available for execution on the device"
# hence the cu128 wheel index below.
#
# No model weights are baked in — they live on the network volume.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# --- torch, CUDA 12.8 ------------------------------------------------------
RUN pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Fail the build here rather than on the rented GPU.
RUN python -c "import torch; \
v = tuple(int(x) for x in (torch.version.cuda or '0').split('.')[:2]); \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert v >= (12, 8), 'torch built for CUDA %s; Blackwell needs >= 12.8' % torch.version.cuda"

# --- ComfyUI ---------------------------------------------------------------
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && pip install -r /opt/ComfyUI/requirements.txt

# --- custom nodes ----------------------------------------------------------
# rgthree for the Power Lora Loader, Manager so you can add nodes from the UI
# (those land on the volume and persist — see entrypoint.sh).
RUN git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git \
        /opt/ComfyUI/custom_nodes/rgthree-comfy \
    && git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager.git \
        /opt/ComfyUI/custom_nodes/ComfyUI-Manager \
    && for d in /opt/ComfyUI/custom_nodes/*/; do \
           if [ -f "$d/requirements.txt" ]; then pip install -r "$d/requirements.txt"; fi; \
       done

RUN pip install huggingface_hub hf_transfer

# --- our bits --------------------------------------------------------------
COPY scripts/ /opt/scripts/
COPY workflows/ /opt/workflows/
RUN chmod +x /opt/scripts/*.sh /opt/scripts/*.py

ENV MODELS_DIR=/workspace/models \
    COMFY_PORT=8188

WORKDIR /opt/ComfyUI
EXPOSE 8188

CMD ["/opt/scripts/entrypoint.sh"]
