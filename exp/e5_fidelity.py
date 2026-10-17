"""E5: perplexity and KL divergence against the BF16 Qwen3.5 base on a fixed wikitext-2 slice.

Step 1 (once per size): `./run.sh ref qwen35-2b-bf16` saves BF16 logits to refs/<size>.kld.
Step 2 (per model): scored against that file. Without a reference only perplexity is reported.
"""
import json
import re

from lib.common import LOGS, REFS, RESULTS, load_models, log, prism_bin, write_json
from lib.server import run_tool

from . import data

CTX, CHUNKS = 1024, 20
NUM = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"


def parse_stats(text):
    """Pull 'Key : value ± err' lines from llama-perplexity's summary blocks, plus the final PPL estimate."""
    out = {}
    m = re.search(r"Final estimate: PPL = " + NUM + r" \+/- " + NUM, text)
    if m:
        out["ppl"], out["ppl_err"] = float(m.group(1)), float(m.group(2))
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9.%()/ ,:Δ_-]+?)\s*:\s*" + NUM + r"(?:\s*±\s*" + NUM + r")?", line)
        if m and ("KLD" in line or "PPL" in line or "top" in line or "Δp" in line or "Cor(" in line):
            key = re.sub(r"\s+", " ", m.group(1)).strip()
            out[key] = float(m.group(2))
            if m.group(3):
                out[key + " ±"] = float(m.group(3))
    return out


def _perplexity(args, log_path, cpu_fallback=True):
    base = [prism_bin("llama-perplexity"), "-c", str(CTX), "-fa", "on", *args]
    rc, text = run_tool(base + ["-ngl", "99"], log_path, timeout=4 * 3600)
    if rc != 0 and cpu_fallback and re.search(r"out of memory|failed to allocate", text, re.I):
        log("E5: GPU OOM, retrying on CPU (-ngl 0)")
        rc, text = run_tool(base + ["-ngl", "0"], log_path, timeout=8 * 3600)
    if rc != 0:
        raise RuntimeError(f"llama-perplexity rc={rc}; see {log_path}")
    return text


def make_ref(ref_id, cfg, model_path):
    REFS.mkdir(exist_ok=True)
    out = REFS / f"{cfg['size']}.kld"
    text = _perplexity(["-m", str(model_path), "-f", str(data.wikitext_raw()), "--chunks", str(CHUNKS),
                        "--kl-divergence-base", str(out)], LOGS / ref_id / "e5_ref.log")
    stats = parse_stats(text)
    write_json(RESULTS / ref_id / "e5_ref.json", {"ref": ref_id, "file": cfg["file"], "ctx": CTX,
                                                  "chunks": CHUNKS, "kld_file_gb": round(out.stat().st_size / 1e9, 2),
                                                  **stats})
    log(f"E5 reference {ref_id}: {stats}")


def run(mid, model_path, outdir, baseline):
    cfg = load_models()[mid]
    ref = REFS / f"{cfg['size']}.kld"
    if cfg.get("ref") and ref.exists():
        text = _perplexity(["-m", str(model_path), "--kl-divergence-base", str(ref), "--kl-divergence"],
                           LOGS / mid / "e5.log")
        mode = "kld"
    else:
        log(f"E5: no reference at {ref}; perplexity only")
        text = _perplexity(["-m", str(model_path), "-f", str(data.wikitext_raw()), "--chunks", str(CHUNKS)],
                           LOGS / mid / "e5.log")
        mode = "ppl_only"
    res = {"model": mid, "mode": mode, "ref": cfg.get("ref"), "ctx": CTX, "chunks": CHUNKS, **parse_stats(text)}
    (outdir / "e5_fidelity.json").write_text(json.dumps(res, indent=2))
    log(f"E5 {mid}: {res}")
    return res
