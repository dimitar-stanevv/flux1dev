# flux-comfyui-runpod

ComfyUI + FLUX.1-dev on RunPod, configured entirely through environment
variables, with all weights living on a network volume so restarts don't
re-download 35 GB.

```
Dockerfile                        image for the RunPod template
.github/workflows/build.yml       CI build + publish to GHCR
scripts/entrypoint.sh             wires ComfyUI to the volume, then launches
scripts/provision.py              idempotent downloader (models + LoRAs)
scripts/bootstrap.sh              alternative: no Docker build, Start command only
workflows/flux_dev_power_lora.json    Flux + rgthree Power Lora Loader
workflows/flux_dev_lora_chain.json    Flux + 3 stock LoraLoaderModelOnly nodes
tools/gen_workflows.py            regenerates + validates those two graphs
.env.example                      every env var, documented
Makefile                          make check / build / run / workflows
```

No LoRA is baked into the image or the workflows — they ship with empty
placeholder slots. Point `LORAS` at your own files and they land on the volume
on the next boot.

## How the caching works

`provision.py` writes `<volume>/models/.provision-manifest.json` recording the
source URL and byte size of everything it fetched. On each boot, for every
entry in `LORAS` + the base model list:

1. file missing → download
2. file present and manifest matches url+size → skip, no network call
3. file present but not in the manifest → `HEAD` the URL, compare
   `Content-Length`, skip if equal
4. sizes differ → re-download

So adding one line to `LORAS` fetches exactly that one file. Downloads go to
`.part` and are renamed on success, so an interrupted pod won't leave a
truncated file that looks valid — and `curl -C -` resumes it.

## Setup

### 1. Build the image

Push to `main` (or run the **build image** workflow by hand) and GitHub Actions
publishes `ghcr.io/dimitar-stanevv/flux1dev:latest`. Nothing to run locally.

**Make the package public once**, or RunPod can't pull it — the package page is
linked from the workflow's summary, under *Package settings → Change visibility*.

> Build in CI rather than locally on Apple Silicon: `--platform linux/amd64`
> runs the torch install under qemu emulation and takes roughly an hour.
> `make build` is there for Linux hosts.

The image is ~9 GB; model weights are deliberately *not* baked in.

### 2. Create the network volume

RunPod → Storage → Network Volume, in a datacenter that has 5090s.
**150 GB minimum** — the fp16 stack is ~35 GB and you'll want room for LoRAs
and outputs.

| file | size |
|---|---|
| flux1-dev.safetensors | ~23.8 GB |
| t5xxl_fp16.safetensors | ~9.8 GB |
| clip_l.safetensors | ~246 MB |
| ae.safetensors | ~335 MB |

### 3. Template

Matching the fields in the Create template dialog:

| field | value |
|---|---|
| Template name | `flux1dev` |
| Template type | Pods |
| Compute type | NVIDIA GPU |
| Container image | `ghcr.io/dimitar-stanevv/flux1dev:latest` |
| Start command | *(leave empty — the image has a CMD)* |
| Container disk | **20 GB** (5 is too small for the venv + torch) |
| Persistent storage / Volume disk | 0 — the network volume overrides this |
| Persistent storage mount path | `/workspace` |
| HTTP Ports | label `comfyui`, port `8188` |
| Environment variables | see `.env.example` |

Put `HF_TOKEN` in as a **Secret** (the key icon), not a plain env var.

### 4. Deploy

Deploy a pod from the template with the network volume attached, on an RTX
5090. First boot downloads ~35 GB (a few minutes on RunPod's network). Open
the pod's `8188` HTTP port → ComfyUI. Load a workflow from the sidebar; both
bundled ones are seeded into `/workspace/user/default/workflows`.

Stop the pod when idle. Restarting skips every download.

## Adding LoRAs

Edit `LORAS` in the template (or the pod's env), restart the pod. New entries
download, existing ones don't. Format, one per line:

```
my_lora.safetensors=https://huggingface.co/your-user/your-repo/resolve/main/my_lora.safetensors?download=true
https://huggingface.co/someone/some-lora/resolve/main/style.safetensors
```

The name on the left is what shows up in the ComfyUI dropdown; a bare URL gets
its name from the URL path. Query strings are fine — the filename ignores
them, and `=` inside a URL doesn't confuse the `name=url` split (it splits on
the first `=` only). Entries separate on newlines, `;`, or a comma that's
directly followed by the next URL.

Then open the workflow and pick your LoRA in the node's dropdown — the bundled
graphs deliberately don't hardcode one.

For private HF repos the same `HF_TOKEN` is used — the `Authorization` header
is only sent to `huggingface.co`, never to third-party hosts.

## The workflows

**`flux_dev_power_lora.json`** — `UNETLoader` → `Power Lora Loader (rgthree)` →
guider/scheduler. The rgthree node holds an arbitrary list of LoRAs with
per-entry on/off toggles and strength sliders; hit **➕ Add Lora** for more
rows. It ships with three slots, all toggled off — pick a LoRA in a row and
flip it on.

rgthree stores its LoRA list in dynamically-created widgets, and that
serialisation has changed between versions. If the node loads with empty
dropdowns, the graph wiring is still correct — just pick your LoRAs from the
dropdowns and re-save. That's the only thing that can go wrong here.

**`flux_dev_lora_chain.json`** — the same graph using three stock
`LoraLoaderModelOnly` nodes in series. No custom node dependency at all. All
three ship **bypassed** (mode 4), so the graph renders base Flux on a fresh
volume without erroring on a LoRA that isn't there. Pick your file in the
dropdown, hit Ctrl+B to un-bypass, set a strength. Slots you don't want stay
bypassed (or at strength 0).

Both use model-only LoRA application, which is right for LoRAs trained with
ai-toolkit / the Replicate trainer — they contain no text-encoder weights. If
you pick up a Civitai LoRA that *does* have CLIP keys, switch that slot to the
full `LoraLoader` and route CLIP through it too.

Defaults: 20 steps, euler/simple, guidance 3.5, 1024×1024. **Put your trigger
word in the prompt** — the seeded prompt starts with `TOK`, which is the
ai-toolkit default; replace it with whatever you trained. A LoRA that seems to
do nothing is almost always a missing trigger word.

To regenerate the graphs after editing `tools/gen_workflows.py`:

```bash
python tools/gen_workflows.py
```

It refuses to write a file whose links don't reconcile from both ends.

Stacking: total strength is what matters, not order — ComfyUI applies the
patches additively. Three LoRAs at 1.0 will over-cook the image; start around
0.6–0.8 each.

## Driving it from your own code

The pod exposes ComfyUI's HTTP API at the same proxy URL:

```
https://<pod-id>-8188.proxy.runpod.net/prompt
```

`POST /prompt` with `{"prompt": <api-format graph>}` queues a job, then poll
`GET /history/<prompt_id>` and fetch results from `GET /view?filename=...`.
To get the API-format graph: in ComfyUI enable **Settings → Enable dev mode
options**, then **Workflow → Export (API)**. That JSON is a dict keyed by node
id — patch the prompt text, seed, and LoRA names in place before posting.

The proxy URL is public and unauthenticated. Anyone with the pod id can queue
jobs on your GPU. Either keep the pod short-lived, or run it with
`--enable-cors-header` behind something of your own. For a persistent
API-style setup, RunPod Serverless with `runpod/worker-comfyui` is the closer
analogue to what Replicate was doing for you.

## No-Docker alternative

If you'd rather not maintain an image: push this repo to GitHub, use a stock
RunPod PyTorch image with CUDA 12.8, and set the Start command to

```
bash -c "curl -fsSL https://raw.githubusercontent.com/dimitar-stanevv/flux1dev/main/scripts/bootstrap.sh | bash"
```

with `REPO_URL` set in the env. It installs ComfyUI, the venv and the models
onto `/workspace`, so only the first boot is slow. Trade-off: it re-checks pip
on every start, and a bad commit breaks every pod immediately.

## Notes

- **Blackwell needs CUDA 12.8+.** Torch built for cu121/cu124 will throw
  `no kernel image is available for execution on the device` on a 5090. The
  Dockerfile pins the cu128 wheel index for this reason.
- FLUX.1-dev is gated — accept the licence on your HF account before the token
  will work, or the downloads 401. Its licence is non-commercial.
- Custom nodes installed through ComfyUI Manager at runtime land in
  `/workspace/custom_nodes` and persist; the entrypoint installs their
  `requirements.txt` on the next boot.
- 32 GB of VRAM fits the whole fp16 stack, so no `--lowvram`, no fp8, no GGUF.
