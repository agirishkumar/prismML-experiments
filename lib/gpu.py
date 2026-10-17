"""nvidia-smi helpers: preflight, VRAM readings and a 100 ms power sampler for energy accounting."""
import csv
import subprocess
import time
from datetime import datetime

from .common import log


def _smi(args):
    return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip()


def gpu_info():
    name, driver, total = [x.strip() for x in _smi(["--query-gpu=name,driver_version,memory.total"]).split(",")]
    return {"gpu": name, "driver": driver, "vram_total_mib": int(total)}


def vram_used():
    return int(_smi(["--query-gpu=memory.used"]))


def compute_apps():
    out = _smi(["--query-compute-apps=pid,process_name,used_memory"])
    apps = []
    for line in filter(None, out.splitlines()):
        pid, name, mem = [x.strip() for x in line.split(",")]
        apps.append({"pid": int(pid), "name": name, "mib": int(mem) if mem.isdigit() else None})
    return apps


def proc_vram(pid):
    for a in compute_apps():
        if a["pid"] == pid:
            return a["mib"]
    return None


def idle_power(seconds=20):
    vals = []
    for _ in range(seconds * 2):
        vals.append(float(_smi(["--query-gpu=power.draw"])))
        time.sleep(0.5)
    return sum(vals) / len(vals)


def preflight():
    apps = compute_apps()
    if apps:
        raise RuntimeError(f"GPU is busy, stop these first: {apps}")
    base = {"vram_used_mib": vram_used(), **gpu_info()}
    log(f"preflight ok: GPU free, baseline VRAM {base['vram_used_mib']} MiB; measuring idle power 20 s")
    base["idle_power_w"] = round(idle_power(), 2)
    return base


COOL_C = 60  # start every experiment / bench test from the same thermal state


def temperature():
    return int(_smi(["--query-gpu=temperature.gpu"]))


def cooldown(target=COOL_C, timeout=900):
    """Block until the GPU is at or below `target` °C. Returns (seconds waited, final temp)."""
    t0, temp = time.time(), temperature()
    if temp > target:
        log(f"cooldown: GPU {temp} °C, waiting for ≤ {target} °C")
    while temp > target and time.time() - t0 < timeout:
        time.sleep(5)
        temp = temperature()
    if temp > target:
        log(f"WARNING: cooldown timed out at {temp} °C")
    return round(time.time() - t0, 1), temp


def wait_vram_back(baseline_mib, tol=150, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        now = vram_used()
        if now <= baseline_mib + tol and not compute_apps():
            return True, now
        time.sleep(2)
    return False, vram_used()


FIELDS = "timestamp,power.draw,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,pstate"


class PowerSampler:
    """Runs `nvidia-smi -lms 100` in the background and writes a CSV for the whole model run."""

    def __init__(self, path):
        self.path, self.proc = path, None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        self.f.write(FIELDS.replace(".", "_") + "\n")
        self.f.flush()
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits",
                                      "-lms", "100"], stdout=self.f, stderr=subprocess.DEVNULL)
        return self

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        if self.proc:
            self.f.close()
        self.proc = None


def load_power(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = datetime.strptime(r["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                rows.append((t, float(r["power_draw"]), float(r["clocks_sm"]), float(r["temperature_gpu"])))
            except (ValueError, KeyError):
                continue
    return rows


def energy_j(samples, t0, t1, idle_w=0.0):
    """Integrate (power - idle) over [t0, t1] using the samples that fall inside the window."""
    pts = [(t, w) for t, w, *_ in samples if t0 <= t <= t1]
    if len(pts) < 2:
        # very short window: use nearest sample * duration
        near = min(samples, key=lambda s: abs(s[0] - t0)) if samples else (t0, 0.0)
        return max(near[1] - idle_w, 0.0) * (t1 - t0)
    j = 0.0
    for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
        j += ((wa + wb) / 2 - idle_w) * (tb - ta)
    # extend to window edges at the boundary power
    j += (pts[0][1] - idle_w) * (pts[0][0] - t0) + (pts[-1][1] - idle_w) * (t1 - pts[-1][0])
    return max(j, 0.0)
