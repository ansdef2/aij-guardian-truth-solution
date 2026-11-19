#!/usr/bin/env bash
# Скачивает веса в solution/model и собирает архив для платформы (Linux).
#   bash tools/run_all.sh                       -> dist/final.zip   (scripts/predict.py)
#   bash tools/run_all.sh predict_border.py     -> dist/final_border.zip
# Если скачивание оборвалось — запустите ту же команду ещё раз: скачанное не перекачивается.
# Долгую сборку на удалённой машине лучше запускать так, чтобы её не убил обрыв связи:
#   nohup bash tools/run_all.sh > build.log 2>&1 &   и смотреть  tail -f build.log
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENTRY="${1:-predict.py}"
NAME="final"; [ "$ENTRY" != "predict.py" ] && NAME="final_$(basename "$ENTRY" .py | sed 's/^predict_//')"
OUT="$REPO/dist/$NAME.zip"

echo "[1/3] Ставлю huggingface_hub..."
python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q -U pip huggingface_hub hf_xet

echo "[2/3] Скачиваю веса Qwen3.6-35B-A3B-FP8 (~35 ГБ) в solution/model ..."
"$REPO/.venv/bin/python" - "$REPO/solution/model" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download("Qwen/Qwen3.6-35B-A3B-FP8", local_dir=sys.argv[1], max_workers=16)
print(f"веса скачаны за {time.time() - t:.0f} с")
PY

echo "[3/3] Собираю $OUT (точка входа: $ENTRY) ..."
"$REPO/.venv/bin/python" "$REPO/tools/build_zip.py" "$REPO/solution" "$OUT" --entry "$ENTRY"
echo "ГОТОВО: $OUT"
