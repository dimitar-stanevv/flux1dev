#!/usr/bin/env python3
"""Idempotent model downloader for the ComfyUI / FLUX RunPod template.

Everything is driven by environment variables set in the RunPod template, so
adding a LoRA never means editing code. Everything lands under $MODELS_DIR
(default /workspace/models) in ComfyUI's standard subdirectories.

The point of this script is what it *doesn't* do: on a restart with an
unchanged config it makes no network calls at all. See `plan_file()` for the
skip rules and `.provision-manifest.json` for the state it keeps.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

MANIFEST_NAME = ".provision-manifest.json"

SUBDIRS = (
    "diffusion_models",
    "clip",
    "vae",
    "loras",
    "controlnet",
    "upscale_models",
    "clip_vision",
    "embeddings",
    "style_models",
)

MODEL_EXTS = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".sft",
    ".gguf",
    ".onnx",
)

FLUX_LICENCE_URL = "https://huggingface.co/black-forest-labs/FLUX.1-dev"

# subdir, filename, url
BASE_MODELS = (
    (
        "diffusion_models",
        "flux1-dev.safetensors",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
    ),
    (
        "vae",
        "ae.safetensors",
        "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
    ),
    (
        "clip",
        "clip_l.safetensors",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    ),
    (
        "clip",
        "t5xxl_fp16.safetensors",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
    ),
)

# Repos that need HF_TOKEN because the licence is click-through gated.
GATED_URL_PREFIXES = ("https://huggingface.co/black-forest-labs/FLUX.1-dev",)


def log(msg=""):
    print("[provision] %s" % msg if msg else "", flush=True)


def env_str(name, default=""):
    return (os.environ.get(name) or default).strip()


def env_bool(name, default=False):
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on")


def human(n):
    if n is None:
        return "unknown"
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < step or unit == "TiB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= step
    return "%.1f TiB" % n


# --------------------------------------------------------------------------
# env parsing
# --------------------------------------------------------------------------

def split_entries(raw):
    """Split a multi-entry env var.

    Separators: newline, ';', or a comma that is immediately followed by a
    URL. A bare comma is *not* a separator — it shows up inside query
    strings often enough that treating it as one breaks real URLs.
    """
    if not raw:
        return []
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r",(?=\s*https?://)", "\n", text)
    text = text.replace(";", "\n")
    out = []
    for line in text.split("\n"):
        line = line.strip().strip('"').strip("'").strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def filename_from_url(url):
    """Filename from the URL *path*, ignoring the query string.

    .../resolve/main/my_lora.safetensors?download=true -> my_lora.safetensors
    """
    path = urlsplit(url).path
    name = unquote(path.rsplit("/", 1)[-1]).strip()
    return os.path.basename(name)


def ensure_ext(name):
    if not name.lower().endswith(MODEL_EXTS):
        name += ".safetensors"
    return name


def parse_lora_entry(entry):
    """'name=url' or a bare url -> (filename, url), or None if unusable."""
    entry = entry.strip()
    if not entry:
        return None

    if entry.lower().startswith(("http://", "https://")):
        # Bare URL. Never split on '=' here: query strings are full of them.
        url = entry
        name = filename_from_url(url)
    else:
        # First '=' only, for the same reason.
        name, sep, url = entry.partition("=")
        if not sep:
            log("skipping unparseable LORAS entry: %r" % entry)
            return None
        name = os.path.basename(unquote(name.strip()))
        url = url.strip()
        if not url.lower().startswith(("http://", "https://")):
            log("skipping LORAS entry with no usable URL: %r" % entry)
            return None
        if not name:
            name = filename_from_url(url)

    if not name:
        log("could not work out a filename for: %r" % entry)
        return None
    return ensure_ext(name), url


def parse_loras(raw):
    out = []
    for entry in split_entries(raw):
        parsed = parse_lora_entry(entry)
        if parsed:
            out.append(("loras", parsed[0], parsed[1]))
    return out


def parse_extra_models(raw):
    """'folder|filename|url' per entry. 'folder|url' also works."""
    out = []
    for entry in split_entries(raw):
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) >= 3:
            folder, name, url = parts[0], parts[1], "|".join(parts[2:])
        elif len(parts) == 2:
            folder, url = parts
            name = filename_from_url(url)
        else:
            log("skipping unparseable EXTRA_MODELS entry: %r" % entry)
            continue
        if not folder or not url.lower().startswith(("http://", "https://")):
            log("skipping unparseable EXTRA_MODELS entry: %r" % entry)
            continue
        name = os.path.basename(unquote(name))
        if not name:
            log("could not work out a filename for: %r" % entry)
            continue
        out.append((os.path.basename(folder), name, url))
    return out


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def load_manifest(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        log("manifest was not an object, starting a fresh one")
    except FileNotFoundError:
        pass
    except (ValueError, OSError) as exc:
        log("could not read manifest (%s), starting a fresh one" % exc)
    return {}


def save_manifest(path, manifest):
    """Write after every file — a killed pod should not lose the record."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        log("WARNING: could not write manifest: %s" % exc)


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

def is_huggingface(url):
    return urlsplit(url).hostname in ("huggingface.co", "cdn-lfs.huggingface.co")


def curl_common(token):
    """Auth goes in over stdin, not argv — argv is world-readable in /proc."""
    if token:
        return ["-K", "-"], 'header = "Authorization: Bearer %s"\n' % token
    return [], None


def remote_size(url, token):
    """Content-Length of the final response, or None if we can't tell."""
    auth, stdin_data = curl_common(token if is_huggingface(url) else "")
    cmd = ["curl", "-sIL", "--max-time", "60"] + auth + ["--", url]
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except OSError as exc:
        log("HEAD failed (%s)" % exc)
        return None
    if proc.returncode != 0:
        return None
    size = None
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                # Last one wins: -L means we see every hop's headers.
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return size


def download(url, dest, token):
    """curl to <dest>.part, then rename. Never leave a truncated real file."""
    part = dest + ".part"
    auth, stdin_data = curl_common(token if is_huggingface(url) else "")

    def run(resume):
        cmd = ["curl", "-fL", "--retry", "5", "--retry-all-errors", "--progress-bar"]
        if resume:
            cmd += ["-C", "-"]
        cmd += auth + ["-o", part, "--", url]
        return subprocess.run(cmd, input=stdin_data, universal_newlines=True).returncode

    rc = run(resume=os.path.exists(part))
    # 33: server has no ranged-request support. 36: bad resume offset.
    if rc in (33, 36) and os.path.exists(part):
        log("resume rejected by the server, restarting this file from zero")
        try:
            os.remove(part)
        except OSError:
            pass
        rc = run(resume=False)

    if rc != 0:
        log("ERROR: curl exited %d for %s" % (rc, url))
        return None
    if not os.path.exists(part):
        log("ERROR: curl reported success but wrote nothing for %s" % url)
        return None

    os.replace(part, dest)
    return os.path.getsize(dest)


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def plan_file(key, path, url, manifest, token, force, skip_size_check):
    """Decide what to do with one file. Returns 'download' or 'skip'."""
    if force:
        return "download"

    if not os.path.exists(path):
        return "download"

    local_size = os.path.getsize(path)
    record = manifest.get(key)

    # Known file, unchanged config: the fast path — no network call at all.
    if record and record.get("url") == url and record.get("size") == local_size:
        return "skip"

    if skip_size_check:
        manifest[key] = {"url": url, "size": local_size}
        return "skip"

    remote = remote_size(url, token)
    if remote is None:
        # Can't tell. Trust what's on disk rather than burn 24 GB of transfer.
        log("  %s: remote size unknown, keeping the local file" % key)
        return "skip"
    if remote == local_size:
        manifest[key] = {"url": url, "size": local_size}
        return "skip"

    log("  %s: local %s vs remote %s — re-downloading"
        % (key, human(local_size), human(remote)))
    return "download"


def prune_loras(models_dir, manifest, wanted_keys):
    """Drop manifest-tracked LoRAs that are no longer listed in LORAS."""
    removed = 0
    for key in sorted(k for k in manifest if k.startswith("loras/")):
        if key in wanted_keys:
            continue
        path = os.path.join(models_dir, key)
        try:
            if os.path.exists(path):
                os.remove(path)
                log("pruned %s" % key)
            manifest.pop(key, None)
            removed += 1
        except OSError as exc:
            log("WARNING: could not prune %s: %s" % (key, exc))
    return removed


# --------------------------------------------------------------------------

def main():
    models_dir = env_str("MODELS_DIR", "/workspace/models")
    manifest_path = os.path.join(models_dir, MANIFEST_NAME)
    token = env_str("HF_TOKEN")

    force = env_bool("FORCE_REDOWNLOAD", False)
    skip_size_check = env_bool("SKIP_SIZE_CHECK", False)
    want_base = env_bool("DOWNLOAD_BASE_MODELS", True)

    log("models dir: %s" % models_dir)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(models_dir, sub), exist_ok=True)

    queue = []
    if want_base:
        queue.extend(BASE_MODELS)
    else:
        log("DOWNLOAD_BASE_MODELS is off — skipping the FLUX base stack")
    loras = parse_loras(env_str("LORAS"))
    queue.extend(loras)
    queue.extend(parse_extra_models(env_str("EXTRA_MODELS")))

    if not token and want_base:
        log()
        log("!" * 70)
        log("HF_TOKEN is empty. flux1-dev.safetensors and ae.safetensors live in")
        log("a gated repo and will 401 without it.")
        log("Accept the licence at %s, create a read token, and set HF_TOKEN as a" % FLUX_LICENCE_URL)
        log("RunPod Secret. Carrying on regardless — the ungated files will work.")
        log("!" * 70)
        log()

    manifest = load_manifest(manifest_path)
    downloaded = skipped = failed = 0
    failures = []
    seen = set()

    for subdir, filename, url in queue:
        key = "%s/%s" % (subdir, filename)
        if key in seen:
            log("%s listed more than once, using the first entry" % key)
            continue
        seen.add(key)

        dest_dir = os.path.join(models_dir, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, filename)

        action = plan_file(key, path, url, manifest, token, force, skip_size_check)
        if action == "skip":
            skipped += 1
            log("skip     %s" % key)
            save_manifest(manifest_path, manifest)
            continue

        log("download %s" % key)
        log("         <- %s" % url)
        size = download(url, path, token)
        if size is None:
            failed += 1
            failures.append(key)
            if url.startswith(GATED_URL_PREFIXES):
                log("         (gated repo — check HF_TOKEN and the licence at %s)"
                    % FLUX_LICENCE_URL)
            continue

        manifest[key] = {"url": url, "size": size}
        save_manifest(manifest_path, manifest)
        downloaded += 1
        log("         ok, %s" % human(size))

    if env_bool("PRUNE_LORAS", False):
        wanted = set("loras/%s" % name for _, name, _ in loras)
        if prune_loras(models_dir, manifest, wanted):
            save_manifest(manifest_path, manifest)

    log()
    log("-" * 60)
    log("downloaded %d   skipped %d   failed %d" % (downloaded, skipped, failed))
    for key in failures:
        log("  FAILED: %s" % key)
    try:
        usage = shutil.disk_usage(models_dir)
        log("volume: %s free of %s" % (human(usage.free), human(usage.total)))
    except OSError as exc:
        log("could not stat the volume: %s" % exc)
    log("-" * 60)

    # Never block ComfyUI from starting: a missing LoRA is a broken node,
    # not a broken pod.
    return 0


if __name__ == "__main__":
    sys.exit(main())
