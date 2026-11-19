"""Сборка архива решения для платформы.

    python tools/build_zip.py solution dist/final.zip
    python tools/build_zip.py solution dist/border.zip --entry predict_border.py

Правила платформы, которые здесь учтены:
  * принимается только zip (tar/tar.gz падают на распаковке);
  * файлы лежат в КОРНЕ архива, без папки верхнего уровня — иначе «can't open file scripts/predict.py»;
  * точка входа — scripts/predict.py; --entry кладёт выбранный скрипт под этим именем;
  * явные права: папки 755, файлы 644;
  * архив меньше 40 ГБ (веса Qwen3.6-35B-A3B-FP8 — ~34.9 ГиБ ≈ 37.5 ГБ).
Веса не сжимаются (ZIP_STORED): safetensors почти не жмутся, а сборка идёт в разы быстрее.
Служебные папки (__pycache__, .git, .cache от huggingface_hub, model_hf, .venv) пропускаются.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", "model_hf", "metrics", ".cache", ".venv", "runs"}
LIMIT = 40 * 10**9


def write_zip(root: Path, out: Path, entry: str = "predict.py", require_weights: bool = True) -> Path:
    root, out = root.resolve(), out.resolve()
    if not (root / "scripts" / entry).exists():
        sys.exit(f"нет точки входа {root / 'scripts' / entry}")
    if not (root / "pyproject.toml").exists():
        sys.exit(f"нет {root / 'pyproject.toml'}")
    weights = list((root / "model").glob("*.safetensors"))
    if require_weights and not weights:
        sys.exit("в model/ нет *.safetensors — сначала скачайте веса (tools/run_all.sh)")
    if weights:
        print(f"весов: {len(weights)} файлов, {sum(f.stat().st_size for f in weights) / 2**30:.1f} ГиБ")

    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            rel_dir = Path(dirpath).relative_to(root)
            if str(rel_dir) != ".":
                info = zipfile.ZipInfo(rel_dir.as_posix() + "/")
                info.external_attr = (0o40755 << 16) | 0x10
                z.writestr(info, b"")
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.resolve() == out:
                    continue
                arc = (rel_dir / fn).as_posix() if str(rel_dir) != "." else fn
                if rel_dir.as_posix() == "scripts" and fn.startswith("predict") and fn.endswith(".py"):
                    if fn != entry:
                        continue                      # другие варианты точки входа не нужны
                    arc = "scripts/predict.py"
                info = zipfile.ZipInfo.from_file(p, arc)
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_STORED if p.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
                with open(p, "rb") as src, z.open(info, "w", force_zip64=True) as dst:
                    while chunk := src.read(64 * 2**20):
                        dst.write(chunk)
                n += 1
                if p.suffix == ".safetensors":
                    print(f"  + {arc} ({p.stat().st_size / 2**30:.2f} ГиБ)", flush=True)
    size = out.stat().st_size
    print(f"готово: {out} — {n} файлов, {size / 2**30:.2f} ГиБ, {time.time() - t0:.0f} с")
    if size >= LIMIT:
        print("ВНИМАНИЕ: архив 40 ГБ или больше — платформа его не примет")
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    bad = [x for x in ("pyproject.toml", "scripts/predict.py") if x not in names]
    print("проверка структуры:", "ок" if not bad else f"нет {bad}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="папка решения (обычно solution)")
    ap.add_argument("out", type=Path, help="куда положить zip")
    ap.add_argument("--entry", default="predict.py", help="какой scripts/*.py станет scripts/predict.py")
    ap.add_argument("--allow-no-weights", action="store_true",
                    help="собрать без весов (только для экспериментов, которые качают модель при запуске)")
    a = ap.parse_args()
    write_zip(a.root, a.out, a.entry, require_weights=not a.allow_no_weights)


if __name__ == "__main__":
    main()
