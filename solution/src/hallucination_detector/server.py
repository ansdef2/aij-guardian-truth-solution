"""Подъём/ожидание/остановка локального vLLM OpenAI-совместимого сервера."""
from __future__ import annotations
import subprocess
import sys
import time
import urllib.request


def start_vllm(model_path, served_name, port, tp=1, max_model_len=131072,
               gpu_util=0.90, extra_args=None):
    """Стартует vLLM в подпроцессе. FP8-квантизация автодетектится из config.json.
    --language-model-only — модель мультимодальная (VL), но нам нужен только текст."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", served_name,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_util),
        "--language-model-only",
        "--enable-prefix-caching",
    ]
    if extra_args:
        cmd += list(extra_args)
    return subprocess.Popen(cmd)


def wait_ready(port, timeout=1200):
    """Ждёт, пока /v1/models начнёт отвечать 200."""
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM not ready on :{port} within {timeout}s")


def stop_vllm(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        proc.kill()
