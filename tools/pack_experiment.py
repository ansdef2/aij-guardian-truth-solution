"""Упаковка эксперимента из experiments/ в архив для платформы (без весов).

    python tools/pack_experiment.py experiments/j3_cascade dist/j3_cascade.zip

Что попадает в архив:
  scripts/predict.py                — experiments/<имя>/predict.py
  src/hallucination_detector/       — модули и промпты из solution/ + experiments/prompts/*.md
                                      + дополнительные *.py из папки эксперимента (например, synthetic.py)
  src/guard/                        — эвристики графа (experiments/graph_signals/guard)
  pyproject.toml, model/README.md

Эксперименты скачивают веса с Hugging Face при запуске. На лидерборде это работало,
но по правилам финальное решение должно нести веса в model/ — поэтому финал собирается
из solution/ через tools/build_zip.py.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from build_zip import write_zip  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path, help="папка эксперимента с predict.py")
    ap.add_argument("out", type=Path, help="куда положить zip")
    a = ap.parse_args()

    exp = a.experiment.resolve()
    if not (exp / "predict.py").exists():
        sys.exit(f"в {exp} нет predict.py")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "submission"
        pkg = root / "src" / "hallucination_detector"
        shutil.copytree(REPO / "solution" / "src" / "hallucination_detector", pkg,
                        ignore=shutil.ignore_patterns("__pycache__"))
        for p in (REPO / "experiments" / "prompts").glob("*.md"):
            shutil.copy(p, pkg / p.name)
        for p in exp.glob("*.py"):
            if p.name != "predict.py":
                shutil.copy(p, pkg / p.name)
        shutil.copytree(REPO / "experiments" / "graph_signals" / "guard", root / "src" / "guard",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (root / "scripts").mkdir(parents=True)
        shutil.copy(exp / "predict.py", root / "scripts" / "predict.py")
        shutil.copy(REPO / "solution" / "pyproject.toml", root / "pyproject.toml")
        (root / "model").mkdir()
        (root / "model" / "README.md").write_text(
            "Эксперимент: веса скачиваются с Hugging Face при запуске.\n", encoding="utf-8")
        write_zip(root, a.out, require_weights=False)


if __name__ == "__main__":
    main()
