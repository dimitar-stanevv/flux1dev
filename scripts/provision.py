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

LICENCE_URL = "https://huggingface.co/black-forest-labs/FLUX.1-dev"

# subdir, filename, url — the full fp16 stack, ~34 GB.
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

# Leave the volume a little room so a full disk doesn't also break ComfyUI's
# own writes (outputs, user settings, Manager installs).
FREE_SPACE_MARGIN = 512 * 1024 * 1024


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


def human_time(seconds):
    if not seconds:
        return "unknown"
    if seconds < 1:
        return "<1s"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "%dh%02dm" % (hours, minutes)
    if minutes:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs


def rate(num_bytes, seconds):
    if not seconds or not num_bytes:
        return "unknown"
    return "%s/s" % human(num_bytes / seconds)


def free_space(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


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


def parse_curl_stats(text):
    """Pull the -w line off curl's stdout: http code, bytes, seconds."""
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return {}
    parts = lines[-1].split()
    if len(parts) != 3:
        return {}
    try:
        return {"code": int(parts[0]), "bytes": float(parts[1]),
                "seconds": float(parts[2])}
    except ValueError:
        return {}


def download(url, dest, token):
    """curl to <dest>.part, then rename. Never leave a truncated real file."""
    part = dest + ".part"
    auth, stdin_data = curl_common(token if is_huggingface(url) else "")
    stats = {}

    def run(resume):
        # curl's *default* progress meter, not --progress-bar: the plain bar
        # shows a percentage and nothing else, which tells you nothing useful
        # when a 24 GB file is crawling. The default meter has live speed and
        # ETA. It goes to stderr; the -w summary comes back on stdout.
        #
        # Plain --retry, NOT --retry-all-errors: the latter retries permanent
        # failures too, so a 403 or a full disk burns five attempts and ~30s
        # of backoff while burying the real cause in warnings. Bare --retry
        # covers only what is actually transient (timeouts, 408, 429, 5xx).
        cmd = ["curl", "-fL", "--retry", "5", "--retry-connrefused",
               "-w", "%{http_code} %{size_download} %{time_total}\n"]
        if resume:
            cmd += ["-C", "-"]
        cmd += auth + ["-o", part, "--", url]
        proc = subprocess.run(cmd, input=stdin_data, stdout=subprocess.PIPE,
                              universal_newlines=True)
        stats.update(parse_curl_stats(proc.stdout))
        return proc.returncode

    rc = run(resume=os.path.exists(part))
    # 33: server has no ranged-request support. 36: bad resume offset.
    if rc in (33, 36) and os.path.exists(part):
        log("         resume rejected by the server, restarting from zero")
        remove_quietly(part)
        rc = run(resume=False)

    if rc != 0:
        explain_failure(rc, stats.get("code"), url, part)
        return None
    if not os.path.exists(part):
        log("ERROR: curl reported success but wrote nothing for %s" % url)
        return None

    os.replace(part, dest)
    return {
        "size": os.path.getsize(dest),
        # Bytes moved *this run* — smaller than size when a .part was resumed.
        "bytes": stats.get("bytes"),
        "seconds": stats.get("seconds"),
    }


def remove_quietly(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except OSError as exc:
        log("WARNING: could not remove %s: %s" % (path, exc))
    return False


def explain_failure(rc, http_code, url, part):
    """Say what actually went wrong, in one line, instead of retry noise."""
    if rc == 23:
        # Write error. On a network volume this is essentially always ENOSPC.
        log("ERROR: out of disk space while writing this file.")
        if remove_quietly(part):
            log("       removed the partial download to free the space back.")
        log("       Grow the network volume, or delete something under "
            "$MODELS_DIR, then restart.")
        return

    if http_code in (401, 403):
        log("ERROR: HTTP %s — authorised but not permitted." % http_code)
        log("       This repo is gated. Accept the licence at %s while logged"
            % LICENCE_URL)
        log("       in as the account owning HF_TOKEN, and make sure the token")
        log("       is a classic Read token (fine-grained tokens need the")
        log("       'public gated repos' scope ticked explicitly).")
        log("       Check which account it is:")
        log("         curl -s -H \"Authorization: Bearer $HF_TOKEN\" "
            "https://huggingface.co/api/whoami-v2")
        # A permanent failure will never resume; don't hoard the bytes.
        remove_quietly(part)
        return

    if http_code == 404:
        log("ERROR: HTTP 404 — no such file. Check the URL: %s" % url)
        remove_quietly(part)
        return

    if http_code and http_code >= 400:
        log("ERROR: HTTP %s for %s" % (http_code, url))
        return

    log("ERROR: curl exited %d for %s" % (rc, url))


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def plan_file(key, path, url, manifest, token, force, skip_size_check):
    """Decide what to do with one file.

    Returns (action, remote_size) where action is 'download' or 'skip'.
    """
    if force:
        return "download", None

    if not os.path.exists(path):
        return "download", None

    local_size = os.path.getsize(path)
    record = manifest.get(key)

    # Known file, unchanged config: the fast path — no network call at all.
    if record and record.get("url") == url and record.get("size") == local_size:
        return "skip", None

    if skip_size_check:
        manifest[key] = {"url": url, "size": local_size}
        return "skip", None

    remote = remote_size(url, token)
    if remote is None:
        # Can't tell. Trust what's on disk rather than burn 24 GB of transfer.
        log("  %s: remote size unknown, keeping the local file" % key)
        return "skip", None
    if remote == local_size:
        manifest[key] = {"url": url, "size": local_size}
        return "skip", None

    log("  %s: local %s vs remote %s — re-downloading"
        % (key, human(local_size), human(remote)))
    return "download", remote


def ensure_room(path, url, token, remote):
    """Refuse to start a download that cannot possibly fit.

    Filling the volume and dying at 60% is the worst outcome: the dead .part
    keeps the space, so every later boot retries and fails a bit faster. If
    the file already on disk is provably the wrong size and there is no room
    for both, drop it first — we have already decided to replace it.
    """
    if remote is None:
        remote = remote_size(url, token)
    if remote is None:
        return True  # unknown size; let curl try

    avail = free_space(os.path.dirname(path))
    if avail is None:
        return True

    need = remote + FREE_SPACE_MARGIN
    if avail >= need:
        return True

    if os.path.exists(path):
        stale = os.path.getsize(path)
        log("         not enough room for both copies (%s free, needs %s)"
            % (human(avail), human(need)))
        log("         the local copy is the wrong size, so removing it first")
        if remove_quietly(path):
            avail = free_space(os.path.dirname(path)) or 0
            if avail >= need:
                return True
            log("         still short after freeing %s" % human(stale))

    log("ERROR: not enough free space. Need %s, have %s."
        % (human(need), human(avail)))
    log("       Grow the network volume or delete something under $MODELS_DIR.")
    return False


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
        log("Accept the licence at %s, create a classic Read token, and set" % LICENCE_URL)
        log("HF_TOKEN as a RunPod Secret. Carrying on — the ungated text encoders")
        log("will still download.")
        log("!" * 70)
        log()

    manifest = load_manifest(manifest_path)
    downloaded = skipped = failed = 0
    total_bytes = total_seconds = 0.0
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

        action, remote = plan_file(key, path, url, manifest, token, force,
                                   skip_size_check)
        if action == "skip":
            skipped += 1
            log("skip     %s" % key)
            save_manifest(manifest_path, manifest)
            continue

        log("download %s" % key)
        log("         <- %s" % url)
        if not ensure_room(path, url, token, remote):
            failed += 1
            failures.append(key)
            continue

        result = download(url, path, token)
        if result is None:
            failed += 1
            failures.append(key)
            continue

        size = result["size"]
        manifest[key] = {"url": url, "size": size}
        save_manifest(manifest_path, manifest)
        downloaded += 1
        moved, secs = result["bytes"], result["seconds"]
        if moved and secs:
            total_bytes += moved
            total_seconds += secs
            log("         ok, %s in %s (%s average)"
                % (human(size), human_time(secs), rate(moved, secs)))
        else:
            log("         ok, %s" % human(size))

    if env_bool("PRUNE_LORAS", False):
        wanted = set("loras/%s" % name for _, name, _ in loras)
        if prune_loras(models_dir, manifest, wanted):
            save_manifest(manifest_path, manifest)

    log()
    log("-" * 60)
    log("downloaded %d   skipped %d   failed %d" % (downloaded, skipped, failed))
    if total_bytes and total_seconds:
        log("transferred %s in %s (%s average)"
            % (human(total_bytes), human_time(total_seconds),
               rate(total_bytes, total_seconds)))
    for key in failures:
        log("  FAILED: %s" % key)
    avail = free_space(models_dir)
    if avail is not None:
        try:
            total = shutil.disk_usage(models_dir).total
            log("volume: %s free of %s" % (human(avail), human(total)))
        except OSError:
            log("volume: %s free" % human(avail))
    log("-" * 60)

    # Never block ComfyUI from starting: a missing LoRA is a broken node,
    # not a broken pod.
    return 0


if __name__ == "__main__":
    sys.exit(main())
