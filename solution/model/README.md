# Веса модели

Сюда кладутся веса **Qwen/Qwen3.6-35B-A3B-FP8** (Apache-2.0, ~34.9 ГиБ, 56 файлов):

```bash
pip install -U huggingface_hub hf_xet
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir solution/model
```

или `bash tools/run_all.sh`, который скачает веса и сразу соберёт архив.
В репозиторий веса не входят (см. `.gitignore`).
