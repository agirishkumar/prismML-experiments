"""Paths, config, logging, downloads and the shared generation settings."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, REFS, RESULTS, LOGS = (ROOT / d for d in ("data", "models", "refs", "results", "logs"))
PRISM = Path.home() / ".local/bonsai2"  # PrismML llama.cpp fork, prism-b10777-e3de715

# One set of generation settings for every model: greedy, thinking off, fixed seed.
GEN = {"temperature": 0.0, "seed": 42, "chat_template_kwargs": {"enable_thinking": False}}
EVAL_CTX = 8192


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prism_env():
    env = os.environ.copy()
    libs = f"{PRISM}:{PRISM / 'cuda12'}"
    env["LD_LIBRARY_PATH"] = libs + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def prism_bin(name):
    return str(PRISM / name)


def runtime_version():
    out = subprocess.run([prism_bin("llama-server"), "--version"], env=prism_env(),
                         capture_output=True, text=True).stderr + ""
    return " ".join(l.strip() for l in out.splitlines() if l.startswith("version"))


def load_models():
    return json.loads((ROOT / "config/models.json").read_text())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def hf_expected_sha(repo, fname):
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main"))
    for x in tree:
        if x["path"] == fname:
            return (x.get("lfs") or {}).get("oid")
    raise FileNotFoundError(f"{fname} not in {repo}")


def hf_download(repo, fname, dest_dir=MODELS):
    """Resumable download with sha256 verification against the HF LFS pointer."""
    dest = Path(dest_dir) / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = hf_expected_sha(repo, fname)
    if dest.exists() and sha256(dest) == expected:
        log(f"have {fname} (sha256 ok)")
        return dest, expected
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    log(f"downloading {repo}/{fname}")
    subprocess.run(["curl", "-fL", "--retry", "3", "-C", "-", "-sS", "-o", str(part), url], check=True)
    got = sha256(part)
    if expected and got != expected:
        part.unlink()
        raise RuntimeError(f"sha256 mismatch for {fname}: {got} != {expected}")
    part.rename(dest)
    return dest, got


def git_commit():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def fail(msg):
    log(f"ERROR: {msg}")
    sys.exit(1)
