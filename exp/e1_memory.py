"""E1: VRAM vs context length and KV-cache type, read from llama-server's own buffer report."""
import csv
import re
import time

from lib.common import LOGS, log
from lib.gpu import proc_vram, vram_used
from lib.server import LlamaServer, StartError

CTXS = [4096, 32768, 131072, 262144]
KV_TYPES = ["f16", "q8_0"]
BUF_RE = re.compile(r"(CUDA\d+|CUDA_Host|CPU(?:_Mapped)?)\s+(model|KV|RS|compute|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
OOM_RE = re.compile(r"out of memory|failed to allocate|cudaMalloc failed|unable to allocate", re.I)


def parse_buffers(text):
    out = {}
    for dev, kind, mib in BUF_RE.findall(text):
        key = f"{'gpu' if dev.startswith('CUDA') and dev != 'CUDA_Host' else 'host'}_{kind.lower()}_mib"
        out[key] = round(out.get(key, 0) + float(mib), 2)
    return out


def run(mid, model_path, outdir, baseline):
    rows = []
    for kv in KV_TYPES:
        for ctx in CTXS:
            lp = LOGS / mid / f"e1_ctx{ctx}_{kv}.log"
            srv = LlamaServer(model_path, lp, ctx=ctx, port=8022, extra=["-ctk", kv, "-ctv", kv, "-lv", "4"])
            row = {"model": mid, "ctx": ctx, "kv_type": kv}
            try:
                srv.start()
                time.sleep(2)
                row.update(status="ok", proc_vram_mib=proc_vram(srv.proc.pid),
                           vram_delta_mib=vram_used() - baseline["vram_used_mib"])
            except StartError as e:
                text = lp.read_text(errors="replace")
                row.update(status="oom" if OOM_RE.search(text) else f"fail: {e}")
            finally:
                text = lp.read_text(errors="replace") if lp.exists() else ""
                srv.stop()
            row.update(parse_buffers(text))
            log(f"E1 ctx={ctx:>6} kv={kv:<4} -> {row.get('status')} proc={row.get('proc_vram_mib')} MiB "
                f"model={row.get('gpu_model_mib')} kv={row.get('gpu_kv_mib')} rs={row.get('gpu_rs_mib')} "
                f"compute={row.get('gpu_compute_mib')}")
            rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "ctx", "kv_type", "status"), k))
    with open(outdir / "e1_memory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows
