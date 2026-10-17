"""E2: burst throughput with llama-bench. Each test runs separately after a GPU cooldown, 3 repetitions.

Laptop GPUs throttle under sustained load, so these are *cool-start* numbers. Sustained speed under
load is recorded per item in E6 / IFEval (server timings) for comparison.
"""
import csv
import json
import time

from lib.common import LOGS, log, prism_bin
from lib.gpu import cooldown, temperature
from lib.server import run_tool

TESTS = [("pp", 512, 0), ("pp", 4096, 0), ("pp", 16384, 0),
         ("tg", 128, 0), ("tg", 128, 4096), ("tg", 128, 16384), ("tg", 128, 32768)]


def run(mid, model_path, outdir, baseline):
    rows = []
    for kind, n, depth in TESTS:
        waited, t_start = cooldown()
        args = ["-p", str(n), "-n", "0"] if kind == "pp" else ["-p", "0", "-n", str(n)]
        lp = LOGS / mid / f"e2_{kind}{n}_d{depth}.log"
        t0 = time.time()
        rc, text = run_tool([prism_bin("llama-bench"), "-m", str(model_path), "-ngl", "99", "-fa", "1",
                             "-r", "3", "-o", "jsonl", "-d", str(depth), *args], lp, timeout=3600)
        t1 = time.time()
        if rc != 0:
            log(f"E2 {kind}{n}@{depth}: llama-bench rc={rc}, see {lp}")
        for line in text.splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                rows.append({"model": mid, "test": f"{kind}{n}", "depth": depth, "tok_s": round(d["avg_ts"], 2),
                             "tok_s_std": round(d["stddev_ts"], 2), "temp_start_c": t_start,
                             "temp_end_c": temperature(), "cooldown_s": waited,
                             "t0": round(t0, 2), "t1": round(t1, 2), "build": d.get("build_commit")})
                r = rows[-1]
                log(f"E2 {r['test']:>8} @depth {depth:>5}: {r['tok_s']:8.1f} ± {r['tok_s_std']:.1f} tok/s "
                    f"({t_start}→{r['temp_end_c']} °C)")
    with open(outdir / "e2_speed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    return rows
