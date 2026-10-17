"""E3: energy accounting. Runs after the power sampler stops; annotates E6 / IFEval summaries.

Reports both raw GPU energy and energy above the idle baseline measured in preflight.
"""
import json

from lib.common import log
from lib.gpu import energy_j, load_power

SOURCES = {"e6": ("e6_bfcl.jsonl", "e6_summary.json", "correct"),
           "ifeval": ("ifeval.jsonl", "ifeval_summary.json", "strict_all")}


def _weighted_tps(rows):
    """Output tokens / generation seconds across all items (server timings), i.e. sustained decode speed."""
    toks = secs = 0.0
    for r in rows:
        if r.get("gen_tps") and r.get("completion_tokens"):
            toks += r["completion_tokens"]
            secs += r["completion_tokens"] / r["gen_tps"]
    return round(toks / secs, 1) if secs else None


def annotate(outdir, idle_w, power_csv, names):
    """Annotate only the experiments of this run, using this run's own power file."""
    if not names or not power_csv.exists():
        return
    samples = load_power(power_csv)
    for name in names:
        rows_f, summ_f, ok_key = SOURCES[name]
        if not (outdir / rows_f).exists() or not (outdir / summ_f).exists():
            continue
        rows = [json.loads(l) for l in (outdir / rows_f).read_text().splitlines()]
        j_raw = j_net = 0.0
        for r in rows:
            r["j_raw"] = round(energy_j(samples, r["t0"], r["t1"]), 3)
            r["j_net"] = round(energy_j(samples, r["t0"], r["t1"], idle_w), 3)
            j_raw += r["j_raw"]
            j_net += r["j_net"]
        (outdir / rows_f).write_text("".join(json.dumps(r) + "\n" for r in rows))
        t0, t1 = rows[0]["t0"], rows[-1]["t1"]
        window = [s for s in samples if t0 <= s[0] <= t1]
        n_ok = sum(bool(r.get(ok_key)) for r in rows)
        toks = sum(r.get("completion_tokens") or 0 for r in rows)
        e = {"seconds": round(t1 - t0, 1), "j_raw": round(j_raw, 1), "j_net": round(j_net, 1),
             "idle_w": idle_w, "avg_power_w": round(sum(s[1] for s in window) / max(len(window), 1), 1),
             "max_temp_c": max((s[3] for s in window), default=None),
             "min_sm_clock_mhz": min((s[2] for s in window), default=None),
             "j_net_per_item": round(j_net / len(rows), 3),
             "j_net_per_correct": round(j_net / n_ok, 3) if n_ok else None,
             "j_net_per_output_token": round(j_net / toks, 4) if toks else None,
             "correct": n_ok, "items": len(rows), "output_tokens": toks,
             "avg_sm_clock_mhz": round(sum(s[2] for s in window) / max(len(window), 1)),
             "pct_time_throttled": round(100 * sum(s[2] < 1200 for s in window) / max(len(window), 1), 1),
             "sustained_gen_tps": _weighted_tps(rows)}
        summ = json.loads((outdir / summ_f).read_text())
        summ["energy"] = e
        (outdir / summ_f).write_text(json.dumps(summ, indent=2))
        log(f"E3 energy [{name}]: {e}")
