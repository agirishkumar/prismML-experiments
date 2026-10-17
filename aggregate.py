#!/usr/bin/env python3
"""Collect results/<model>/ into results/SUMMARY.md comparison tables."""
import csv
import json

from lib.common import RESULTS, load_models

ORDER = ["bonsai2-2b", "qwen35-2b", "bonsai2-4b", "qwen35-4b", "bonsai2-9b"]


def _json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _csv(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def _latest_manifest(d):
    ms = sorted((d / "manifests").glob("*.json")) if (d / "manifests").exists() else []
    full = [m for m in ms if not _json(m).get("limit")]
    return _json(full[-1]) if full else None


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    models = load_models()
    present = [m for m in ORDER if (RESULTS / m).exists()]
    md = ["# Results summary", ""]

    md += ["## Models", table(["model", "file", "size MB", "sha256"],
                              [[m, (mf := _latest_manifest(RESULTS / m) or {}).get("model_file"),
                                mf.get("size_mb"), (mf.get("sha256") or "")[:12]] for m in present]), ""]

    rows = []
    for m in present:
        for r in _csv(RESULTS / m / "e1_memory.csv"):
            rows.append([m, r["ctx"], r["kv_type"], r["status"], r.get("proc_vram_mib"), r.get("gpu_model_mib"),
                         r.get("gpu_kv_mib"), r.get("gpu_rs_mib"), r.get("gpu_compute_mib")])
    md += ["## E1 memory (MiB)", table(["model", "ctx", "kv", "status", "process VRAM", "weights", "KV",
                                       "recurrent", "compute"], rows), ""]

    rows = [[m, r["test"], r["depth"], r["tok_s"], r["tok_s_std"]] for m in present
            for r in _csv(RESULTS / m / "e2_speed.csv")]
    md += ["## E2 speed (tok/s)", table(["model", "test", "depth", "tok/s", "±"], rows), ""]

    rows = []
    for m in present:
        s = _json(RESULTS / m / "e6_summary.json")
        if s:
            e = s.get("energy", {})
            rows.append([m, s["simple"]["accuracy"], s["multiple"]["accuracy"], s["irrelevance"]["accuracy"],
                         s["simple"]["items_with_extra_calls"] + s["multiple"]["items_with_extra_calls"],
                         e.get("j_net_per_correct"), e.get("avg_power_w"), e.get("max_temp_c")])
    md += ["## E6 tool calling (BFCL v3) + E3 energy",
           table(["model", "simple", "multiple", "irrelevance (restraint)", "items w/ extra calls",
                  "J per correct", "avg W", "max °C"], rows), ""]

    rows = []
    for m in present:
        s = _json(RESULTS / m / "ifeval_summary.json")
        if s:
            e = s.get("energy", {})
            rows.append([m, s["strict_prompt"], s["strict_instruction"], s["loose_prompt"], s["loose_instruction"],
                         s["mean_completion_tokens"], s["truncated_rate"], s["loop_rate"],
                         e.get("j_net_per_correct"), e.get("j_net_per_output_token")])
    md += ["## IFEval + E3 energy",
           table(["model", "strict prompt", "strict instr", "loose prompt", "loose instr", "mean out tokens",
                  "truncated", "loop rate", "J per correct (strict)", "J per out token"], rows), ""]

    rows = []
    for m in present:
        s = _json(RESULTS / m / "e5_fidelity.json")
        if s:
            rows.append([m, s["mode"], s.get("ppl") or s.get("Mean PPL(Q)"), s.get("Mean KLD"),
                         s.get("99.0% KLD"), s.get("Same top p")])
    md += ["## E5 fidelity vs Qwen3.5 BF16 (wikitext-2, 20×1024)",
           table(["model", "mode", "PPL", "mean KLD", "p99 KLD", "same top-1 %"], rows), ""]

    for ref in [k for k, v in models.items() if v.get("role") == "reference"]:
        s = _json(RESULTS / ref / "e5_ref.json")
        if s:
            md += [f"Reference {ref}: PPL {s.get('ppl')}", ""]

    (RESULTS / "SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
