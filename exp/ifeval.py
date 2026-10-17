"""IFEval: all 541 prompts, thinking off, scored with Google's official instruction checkers (vendored)."""
import json
import sys
import time

from lib.common import DATA, GEN, LOGS, ROOT, log
from lib.server import LlamaServer

from . import data

sys.path.insert(0, str(ROOT / "third_party"))
import nltk  # noqa: E402

nltk.data.path.insert(0, str(DATA / "nltk_data"))
if not (DATA / "nltk_data" / "tokenizers" / "punkt").exists():
    nltk.download("punkt", download_dir=str(DATA / "nltk_data"), quiet=True)
from instruction_following_eval import evaluation_lib as ev  # noqa: E402

MAX_TOKENS = 2048


def dup_4gram_ratio(text):
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not grams else round(1 - len(set(grams)) / len(grams), 4)


def run(mid, model_path, outdir, baseline):
    items = data.ifeval()
    rows = []
    with LlamaServer(model_path, LOGS / mid / "ifeval_server.log") as srv:
        for i, it in enumerate(items):
            payload = {"messages": [{"role": "user", "content": it["prompt"]}], "max_tokens": MAX_TOKENS, **GEN}
            t0 = time.time()
            resp = srv.chat(payload)
            t1 = time.time()
            ch = resp["choices"][0]
            u = resp.get("usage", {})
            text = ch["message"].get("content") or ""
            rows.append({"key": it["key"], "response": text, "finish": ch.get("finish_reason"),
                         "reasoning_leak": bool(ch["message"].get("reasoning_content")),
                         "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                         "dup4": dup_4gram_ratio(text), "t0": t0, "t1": t1,
                         "gen_tps": (resp.get("timings") or {}).get("predicted_per_second"),
                         "pp_tps": (resp.get("timings") or {}).get("prompt_per_second")})
            if (i + 1) % 50 == 0:
                log(f"IFEval {i + 1}/{len(items)}")
    by_key = {r["key"]: r for r in rows}
    p2r = {it["prompt"]: by_key[it["key"]]["response"] for it in items}
    agg = {"strict": [0, 0, 0, 0], "loose": [0, 0, 0, 0]}  # prompt_ok, prompts, instr_ok, instrs
    for it in items:
        inp = ev.InputExample(key=it["key"], instruction_id_list=it["instruction_id_list"],
                              prompt=it["prompt"], kwargs=it["kwargs"])
        r = by_key[it["key"]]
        for mode, fn in (("strict", ev.test_instruction_following_strict), ("loose", ev.test_instruction_following_loose)):
            out = fn(inp, p2r)
            r[f"{mode}_all"] = out.follow_all_instructions
            r[f"{mode}_list"] = out.follow_instruction_list
            a = agg[mode]
            a[0] += out.follow_all_instructions
            a[1] += 1
            a[2] += sum(out.follow_instruction_list)
            a[3] += len(out.follow_instruction_list)
    with open(outdir / "ifeval.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    summary = {f"{m}_{lvl}": round(a[0] / a[1] if lvl == "prompt" else a[2] / a[3], 4)
               for m, a in agg.items() for lvl in ("prompt", "instruction")}
    summary.update(n=n, max_tokens=MAX_TOKENS,
                   mean_completion_tokens=round(sum(r["completion_tokens"] or 0 for r in rows) / n, 1),
                   truncated_rate=round(sum(r["finish"] == "length" for r in rows) / n, 4),
                   loop_rate=round(sum(r["dup4"] > 0.3 and len(r["response"].split()) > 100 for r in rows) / n, 4),
                   reasoning_leaks=sum(r["reasoning_leak"] for r in rows))
    (outdir / "ifeval_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"IFEval: {summary}")
    return summary
