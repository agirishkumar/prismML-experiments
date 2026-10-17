"""Start/stop llama-server (Prism fork) and track every child process so cleanup can kill exactly what we started."""
import os
import signal
import socket
import subprocess
import time

import requests

from .common import EVAL_CTX, log, prism_bin, prism_env

_LIVE = set()  # Popen objects we started and have not reaped


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def kill_proc(p, grace=15):
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)
    _LIVE.discard(p)


def kill_all():
    for p in list(_LIVE):
        kill_proc(p)


def run_tool(args, log_path, timeout=None):
    """Run a Prism CLI tool (llama-bench, llama-perplexity) to completion, tracked for cleanup."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=prism_env(), start_new_session=True)
        _LIVE.add(p)
        try:
            rc = p.wait(timeout=timeout)
        finally:
            kill_proc(p)
    return rc, log_path.read_text(errors="replace")


class StartError(RuntimeError):
    pass


class LlamaServer:
    def __init__(self, model, log_path, ctx=EVAL_CTX, port=8021, extra=()):
        self.model, self.log_path, self.ctx, self.port, self.extra = model, log_path, ctx, port, list(extra)
        self.proc = None
        self.url = f"http://127.0.0.1:{port}"

    @property
    def args(self):
        return [prism_bin("llama-server"), "-m", str(self.model), "-ngl", "99", "-fa", "on",
                "-c", str(self.ctx), "-np", "1", "--reasoning", "off", "--no-webui",
                "--host", "127.0.0.1", "--port", str(self.port), *self.extra]

    def start(self, timeout=240):
        if not port_free(self.port):
            raise StartError(f"port {self.port} already in use")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.args, stdout=self._f, stderr=subprocess.STDOUT,
                                     env=prism_env(), start_new_session=True)
        _LIVE.add(self.proc)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise StartError(f"llama-server exited rc={self.proc.returncode}")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise StartError("llama-server health timeout")

    def log_text(self):
        return self.log_path.read_text(errors="replace")

    def stop(self):
        if self.proc:
            kill_proc(self.proc)
            self._f.close()
            self.proc = None
        t0 = time.time()
        while not port_free(self.port) and time.time() - t0 < 30:
            time.sleep(0.5)

    def chat(self, payload, timeout=900):
        r = requests.post(self.url + "/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        log("server stopped")
