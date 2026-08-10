#!/usr/bin/env bash
# No-Docker path: run this as the RunPod "Start command" on a stock PyTorch
# image with CUDA 12.8. It installs ComfyUI, a venv and the models onto the
# network volume, so only the first boot is slow.
#
#   bash -c "curl -fsSL https://raw.githubusercontent.com/<you>/<repo>/main/scripts/bootstrap.sh | bash"
#
# Trade-off vs the Dockerfile: pip gets re-checked on every start, and a bad
# commit on main breaks every pod immediately. The image is the safer route.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MODELS_DIR="${MODELS_DIR:-$WORKSPACE/models}"
COMFY_DIR="${COMFY_DIR:-$WORKSPACE/ComfyUI}"
VENV_DIR="${VENV_DIR:-$WORKSPACE/venv}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/template-repo}"
REPO_URL="${REPO_URL:-}"
COMFY_PORT="${COMFY_PORT:-8188}"

log() { printf '[bootstrap] %s\n' "$*"; }

if ! awk -v p="$WORKSPACE" '$2 == p { found = 1 } END { exit !found }' /proc/mounts; then
    log "WARNING: $WORKSPACE is not a mountpoint — nothing here will persist."
fi

command -v git >/dev/null 2>&1 || { apt-get update && apt-get install -y --no-install-recommends git curl; }

# --- venv ------------------------------------------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# --- torch (Blackwell needs cu128) -----------------------------------------
if ! python -c "import torch, sys; v = tuple(int(x) for x in (torch.version.cuda or '0').split('.')[:2]); sys.exit(0 if v >= (12, 8) else 1)" 2>/dev/null; then
    log "installing torch for CUDA 12.8"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

# --- ComfyUI ---------------------------------------------------------------
if [ ! -d "$COMFY_DIR/.git" ]; then
    log "cloning ComfyUI to $COMFY_DIR"
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi
pip install -r "$COMFY_DIR/requirements.txt"
pip install huggingface_hub hf_transfer

mkdir -p "$COMFY_DIR/custom_nodes"
clone_node() {
    local url="$1" dir="$COMFY_DIR/custom_nodes/$2"
    [ -d "$dir/.git" ] || git clone --depth 1 "$url" "$dir"
    [ -f "$dir/requirements.txt" ] && pip install -r "$dir/requirements.txt"
    return 0
}
clone_node https://github.com/rgthree/rgthree-comfy.git rgthree-comfy
clone_node https://github.com/Comfy-Org/ComfyUI-Manager.git ComfyUI-Manager

# --- this repo, for provision.py and the workflows -------------------------
if [ -n "$REPO_URL" ]; then
    if [ -d "$REPO_DIR/.git" ]; then
        git -C "$REPO_DIR" pull --ff-only || log "WARNING: could not update $REPO_DIR"
    else
        git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    fi
    mkdir -p /opt/scripts /opt/workflows
    cp "$REPO_DIR"/scripts/*.py /opt/scripts/ 2>/dev/null || true
    cp "$REPO_DIR"/workflows/*.json /opt/workflows/ 2>/dev/null || true
else
    log "REPO_URL is not set — provision.py and the bundled workflows are unavailable"
fi

mkdir -p "$WORKSPACE/output" "$WORKSPACE/input" "$WORKSPACE/user/default/workflows" "$MODELS_DIR"

cat > "$COMFY_DIR/extra_model_paths.yaml" <<YAML
runpod:
    base_path: $MODELS_DIR
    is_default: true
    checkpoints: checkpoints
    diffusion_models: diffusion_models
    unet: diffusion_models
    clip: clip
    text_encoders: clip
    clip_vision: clip_vision
    vae: vae
    loras: loras
    controlnet: controlnet
    upscale_models: upscale_models
    embeddings: embeddings
    style_models: style_models
YAML

shopt -s nullglob
for wf in /opt/workflows/*.json; do
    dest="$WORKSPACE/user/default/workflows/$(basename "$wf")"
    [ -e "$dest" ] || cp "$wf" "$dest"
done
shopt -u nullglob

if [ -f /opt/scripts/provision.py ]; then
    python /opt/scripts/provision.py || log "WARNING: provisioning failed, starting anyway"
fi

cd "$COMFY_DIR"
log "starting ComfyUI on 0.0.0.0:$COMFY_PORT"
# shellcheck disable=SC2086
exec python main.py \
    --listen 0.0.0.0 \
    --port "$COMFY_PORT" \
    --output-directory "$WORKSPACE/output" \
    --input-directory "$WORKSPACE/input" \
    --user-directory "$WORKSPACE/user" \
    ${COMFY_ARGS:-}
