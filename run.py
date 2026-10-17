#!/usr/bin/env python3
"""Run experiments for ONE model at a time on a 6 GB GPU, then clean up.

  ./run.sh <model-id> [--exp e1,e2,e6,ifeval,e5] [--keep]
  ./run.sh ref <ref-id>          # build the BF16 KL-divergence reference for E5, then delete the BF16 model

E3 (energy) is not a separate step: a 100 ms power sampler runs for the whole model run and
E6 / IFEval integrate it per item.
"""
import argparse
import os
import importlib
import signal
import sys
import time

from exp import energy
from lib import gpu
from lib.common import LOGS, RESULTS, git_commit, hf_download, load_models, log, runtime_version, write_json
from lib.server import LlamaServer, StartError, kill_all

ALL_EXPS = ["e1", "e2", "e6", "ifeval", "e5"]
MODULES = {"e1": "exp.e1_memory", "e2": "exp.e2_speed", "e6": "exp.e6_bfcl", "ifeval": "exp.ifeval",
           "e5": "exp.e5_fidelity"}


def _sigterm(*_):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref_id", nargs="?")
    ap.add_argument("--exp", default=",".join(ALL_EXPS))
    ap.add_argument("--keep", action="store_true", help="keep the model file after the run")
    ap.add_argument("--limit", type=int, help="smoke test: only the first N items per dataset")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _sigterm)
    if a.limit:
        os.environ["PRISM_LIMIT"] = str(a.limit)

    models = load_models()
    if a.model == "ref":
        return build_ref(models, a.ref_id, a.keep)
    if a.model not in models:
        sys.exit(f"unknown model {a.model}; choose from {list(models)}")
    cfg = models[a.model]
    exps = [e.strip() for e in a.exp.split(",") if e.strip()]
    outdir = RESULTS / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    (LOGS / a.model).mkdir(parents=True, exist_ok=True)

    baseline = gpu.preflight()
    manifest = {"model": a.model, **cfg, "experiments": exps, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "runtime": runtime_version(), "repo_commit": git_commit(), "baseline": baseline}
    run_id = time.strftime("%Y%m%d-%H%M%S")
    power_csv = outdir / "raw" / f"power_{run_id}.csv"
    manifest["power_csv"] = power_csv.name
    sampler = gpu.PowerSampler(power_csv)
    model_path = None
    status = "ok"
    try:
        model_path, digest = hf_download(cfg["repo"], cfg["file"])
        if cfg.get("fallback_file") and not fits(a.model, model_path):
            log(f"{cfg['file']} does not fit at eval context; falling back to {cfg['fallback_file']}")
            model_path.unlink()
            model_path, digest = hf_download(cfg["repo"], cfg["fallback_file"])
            manifest["fallback_used"] = True
        manifest.update(model_file=model_path.name, sha256=digest, size_mb=round(model_path.stat().st_size / 1e6, 1))
        sampler.start()
        for e in exps:
            log(f"==== {a.model}: {e}")
            manifest.setdefault("cooldown", {})[e] = gpu.cooldown()
            t0 = time.time()
            importlib.import_module(MODULES[e]).run(a.model, model_path, outdir, baseline)
            manifest.setdefault("timings_s", {})[e] = round(time.time() - t0, 1)
    except KeyboardInterrupt:
        status = "interrupted"
        log("interrupted, cleaning up")
    except Exception as ex:  # noqa: BLE001 — record and still clean up
        status = f"error: {type(ex).__name__}: {ex}"
        log(status)
        raise
    finally:
        manifest["status"] = status
        manifest["cleanup"] = cleanup(sampler, baseline, model_path, a.keep)
        manifest["limit"] = a.limit
        energy.annotate(outdir, baseline["idle_power_w"], power_csv, [e for e in exps if e in energy.SOURCES])
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        mpath = outdir / "manifests" / f"{run_id}_{'-'.join(exps)}.json"
        write_json(mpath, manifest)
        log(f"manifest written: {mpath}")


def fits(mid, model_path):
    """Can the model load at the eval context (8192) with f16 KV on this GPU?"""
    srv = LlamaServer(model_path, LOGS / mid / "fit_check.log", port=8023)
    try:
        srv.start()
        return True
    except StartError:
        return False
    finally:
        srv.stop()


def cleanup(sampler, baseline, model_path, keep):
    kill_all()
    sampler.stop()
    ok, now = gpu.wait_vram_back(baseline["vram_used_mib"])
    res = {"vram_after_mib": now, "vram_back_to_baseline": ok, "model_deleted": False}
    if not ok:
        log(f"WARNING: VRAM {now} MiB, baseline {baseline['vram_used_mib']} MiB; apps: {gpu.compute_apps()}")
    if model_path and model_path.exists() and not keep:
        model_path.unlink()
        res["model_deleted"] = True
    log(f"cleanup: {res}")
    return res


def build_ref(models, ref_id, keep):
    cfg = models[ref_id]
    baseline = gpu.preflight()
    path = None
    try:
        path, _ = hf_download(cfg["repo"], cfg["file"])
        importlib.import_module("exp.e5_fidelity").make_ref(ref_id, cfg, path)
    finally:
        kill_all()
        if path and path.exists() and not keep:
            path.unlink()
        ok, now = gpu.wait_vram_back(baseline["vram_used_mib"])
        log(f"ref cleanup: vram {now} MiB back={ok}, model deleted={not keep}")


if __name__ == "__main__":
    main()
