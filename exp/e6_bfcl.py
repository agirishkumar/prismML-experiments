"""E6: tool calling on BFCL v3 simple / multiple / irrelevance through the OpenAI-compatible endpoint.

Scoring re-implements BFCL's AST checker rules (function name, required params, no unknown params,
type check, value in the allowed list, BFCL string normalisation). Irrelevance passes only when the
model makes no tool call. Same prompts, tools and decoding for every model.
"""
import json
import re
import time

from lib.common import GEN, LOGS, log
from lib.server import LlamaServer

from . import data

CATEGORIES = ["simple", "multiple", "irrelevance"]
TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


# ---------- tool schema conversion (BFCL -> OpenAI tools) ----------
def _schema(s):
    s = dict(s)
    if "type" in s:
        s["type"] = TYPE_MAP.get(s["type"], s["type"])
    if "properties" in s:
        s["properties"] = {k: _schema(v) for k, v in s["properties"].items()}
    if "items" in s:
        s["items"] = _schema(s["items"])
    return s


def to_tools(functions):
    tools, names = [], {}
    for fn in functions:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", fn["name"])
        names[safe] = fn["name"]
        tools.append({"type": "function", "function": {"name": safe, "description": fn.get("description", ""),
                                                       "parameters": _schema(fn["parameters"])}})
    return tools, names


# ---------- BFCL-style AST checking ----------
def _norm(s):
    return re.sub(r"[ ,./\-_*^]", "", s).lower()


def _eq(val, cand):
    if isinstance(cand, str) and isinstance(val, str):
        return _norm(val) == _norm(cand)
    if isinstance(cand, bool) or isinstance(val, bool):
        return val is cand
    if isinstance(cand, (int, float)) and isinstance(val, (int, float)):
        return float(val) == float(cand)
    if isinstance(cand, list) and isinstance(val, (list, tuple)):
        return len(val) == len(cand) and all(_eq(v, c) for v, c in zip(val, cand))
    if isinstance(cand, dict) and isinstance(val, dict):
        for k, allowed in cand.items():
            if k in val:
                if not any(a != "" and _eq(val[k], a) for a in allowed):
                    return False
            elif "" not in allowed:
                return False
        return all(k in cand for k in val)
    return val == cand


def _type_ok(val, t):
    return {"integer": isinstance(val, int) and not isinstance(val, bool),
            "float": isinstance(val, (int, float)) and not isinstance(val, bool),
            "number": isinstance(val, (int, float)) and not isinstance(val, bool),
            "string": isinstance(val, str), "boolean": isinstance(val, bool),
            "array": isinstance(val, list), "tuple": isinstance(val, list),
            "dict": isinstance(val, dict), "object": isinstance(val, dict)}.get(t, True)


def check_call(call, functions, ground_truth):
    """call = (bfcl_name, args). ground_truth = [{fn_name: {param: [allowed...]}}] with one entry."""
    name, args = call
    gt_name, gt_params = next(iter(ground_truth[0].items()))
    if name != gt_name:
        return False, f"wrong function {name} (expected {gt_name})"
    spec = next(f for f in functions if f["name"] == name)["parameters"]
    props, required = spec.get("properties", {}), spec.get("required", [])
    for r in required:
        if r not in args:
            return False, f"missing required param {r}"
    for k, v in args.items():
        if k not in props:
            return False, f"unexpected param {k}"
        if not _type_ok(v, props[k].get("type")):
            return False, f"type mismatch for {k}"
        if k in gt_params and not any(a != "" and _eq(v, a) for a in gt_params[k]):
            return False, f"wrong value for {k}: {v!r}"
    for k, allowed in gt_params.items():
        if k not in args and "" not in allowed:
            return False, f"missing param {k}"
    return True, "ok"


def parse_calls(msg, names):
    calls, errors = [], []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            errors.append("invalid JSON arguments")
            continue
        calls.append((names.get(fn.get("name"), fn.get("name")), args))
    content = msg.get("content") or ""
    unparsed = "<tool_call>" in content or "<function=" in content
    return calls, errors, unparsed


def run(mid, model_path, outdir, baseline):
    rows = []
    with LlamaServer(model_path, LOGS / mid / "e6_server.log") as srv:
        for cat in CATEGORIES:
            for item, gt in data.bfcl(cat):
                tools, names = to_tools(item["function"])
                payload = {"messages": item["question"][0], "tools": tools, "max_tokens": 512, **GEN}
                t0 = time.time()
                resp = srv.chat(payload)
                t1 = time.time()
                choice = resp["choices"][0]
                msg = choice["message"]
                calls, errors, unparsed = parse_calls(msg, names)
                n_calls = len(calls) + len(errors) + (1 if unparsed and not calls else 0)
                if cat == "irrelevance":
                    correct, reason = n_calls == 0, "ok" if n_calls == 0 else "called a tool when none fit"
                elif errors or (unparsed and not calls):
                    correct, reason = False, "unparseable tool call"
                elif len(calls) != 1:
                    correct, reason = False, f"{len(calls)} calls (expected 1)"
                else:
                    correct, reason = check_call(calls[0], item["function"], gt)
                u = resp.get("usage", {})
                rows.append({"id": item["id"], "category": cat, "correct": correct, "reason": reason,
                             "n_calls": n_calls, "extra_calls": max(n_calls - (0 if cat == "irrelevance" else 1), 0),
                             "calls": [[n, a] for n, a in calls], "content": (msg.get("content") or "")[:500],
                             "reasoning_leak": bool(msg.get("reasoning_content")),
                             "finish": choice.get("finish_reason"), "prompt_tokens": u.get("prompt_tokens"),
                             "completion_tokens": u.get("completion_tokens"), "t0": t0, "t1": t1,
                             "gen_tps": (resp.get("timings") or {}).get("predicted_per_second"),
                             "pp_tps": (resp.get("timings") or {}).get("prompt_per_second")})
            done = [r for r in rows if r["category"] == cat]
            log(f"E6 {cat:<11}: {sum(r['correct'] for r in done)}/{len(done)} correct")
    with open(outdir / "e6_bfcl.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {}
    for cat in CATEGORIES:
        rs = [r for r in rows if r["category"] == cat]
        summary[cat] = {"n": len(rs), "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
                        "extra_calls": sum(r["extra_calls"] for r in rs),
                        "items_with_extra_calls": sum(r["extra_calls"] > 0 for r in rs),
                        "reasoning_leaks": sum(r["reasoning_leak"] for r in rs)}
    (outdir / "e6_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
