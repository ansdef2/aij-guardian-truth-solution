#!/usr/bin/env python
"""Замер скорости скачивания весов с Hugging Face на машине платформы.
Скачивает один шард модели бейзлайна, печатает скорость и прогноз времени на всю модель,
затем НАМЕРЕННО завершается с ошибкой (попытка возвращается, лог доступен).
"""
import os
import subprocess
import sys
import tempfile
import time

REPO = "Qwen/Qwen3.6-35B-A3B-FP8"

CHILD = r'''
import os, sys, time
from huggingface_hub import HfApi, hf_hub_download
repo, fname, size, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
t0 = time.time()
hf_hub_download(repo, fname, local_dir=d)
dt = time.time() - t0
print(f"скачано {size/2**30:.2f} ГБ за {dt:.1f} с -> {size/2**20/dt:.0f} МБ/с", flush=True)
'''


def main():
    print("python", sys.version.split()[0], flush=True)
    for mod in ["huggingface_hub", "hf_xet", "hf_transfer"]:
        try:
            m = __import__(mod)
            print(f"{mod}: {getattr(m, '__version__', 'есть')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{mod}: нет ({type(e).__name__})", flush=True)

    from huggingface_hub import HfApi
    info = HfApi().model_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    total = sum(sz for _, sz in files)
    shards = sorted([f for f in files if f[0].endswith(".safetensors")], key=lambda x: -x[1])
    print(f"{REPO}: файлов {len(files)}, всего {total / 2**30:.1f} ГБ, шардов {len(shards)}, "
          f"крупнейший {shards[0][1] / 2**30:.2f} ГБ", flush=True)

    # три способа: обычный, через hf_transfer, через hf_xet отключённый — по одному шарду на способ
    variants = [("по умолчанию", {}),
                ("HF_HUB_ENABLE_HF_TRANSFER=1", {"HF_HUB_ENABLE_HF_TRANSFER": "1"}),
                ("HF_HUB_DISABLE_XET=1", {"HF_HUB_DISABLE_XET": "1"})]
    for k, (name, extra) in enumerate(variants):
        fname, size = shards[k % len(shards)]
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, **extra)
            t0 = time.time()
            r = subprocess.run([sys.executable, "-c", CHILD, REPO, fname, str(size), d], env=env,
                               capture_output=True, text=True, timeout=600)
            out = (r.stdout + r.stderr).strip().splitlines()
            print(f"[{name}] {fname}: {out[-1] if out else ''} (код {r.returncode}, {time.time() - t0:.0f} с)",
                  flush=True)
            if r.returncode == 0 and "МБ/с" in (out[-1] if out else ""):
                mbps = float(out[-1].split("->")[1].split()[0])
                print(f"    прогноз на всю модель: {total / 2**20 / mbps / 60:.1f} мин", flush=True)

    print("конец замера", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
