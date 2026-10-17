"""Download and cache the public datasets (pinned to the files listed here)."""
import json
import os
import urllib.request

from lib.common import DATA, log

BFCL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/"
IFEVAL = "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"
WIKI = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"


def _get(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"fetching {url.rsplit('/', 1)[-1]}")
        urllib.request.urlretrieve(url, dest)
    return dest


def _jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _limit(items):
    n = int(os.environ.get("PRISM_LIMIT", 0))
    return items[:n] if n else items


def bfcl(category):
    """Returns [(item, possible_answer_or_None)] for simple / multiple / irrelevance."""
    items = _jsonl(_get(BFCL + f"BFCL_v3_{category}.json", DATA / "bfcl" / f"BFCL_v3_{category}.json"))
    if category == "irrelevance":
        return _limit([(it, None) for it in items])
    ans = _jsonl(_get(BFCL + f"possible_answer/BFCL_v3_{category}.json",
                      DATA / "bfcl" / "possible_answer" / f"BFCL_v3_{category}.json"))
    by_id = {a["id"]: a["ground_truth"] for a in ans}
    return _limit([(it, by_id[it["id"]]) for it in items])


def ifeval():
    return _limit(_jsonl(_get(IFEVAL, DATA / "ifeval_input_data.jsonl")))


def wikitext_raw():
    """wikitext-2-raw test split joined into the standard wiki.test.raw text file."""
    out = DATA / "wiki.test.raw"
    if not out.exists():
        import pyarrow.parquet as pq
        pqf = _get(WIKI, DATA / "wikitext-2-raw-test.parquet")
        out.write_text("".join(pq.read_table(pqf).column("text").to_pylist()))
    return out
