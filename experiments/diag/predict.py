#!/usr/bin/env python
"""Диагностический сабмит: печатает сведения об окружении платформы и НАМЕРЕННО завершается
с ошибкой, чтобы платформа показала лог (упавшая попытка возвращается).
Предсказаний не делает. Секреты и содержимое данных не печатает.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()[:3000]
    except Exception as e:  # noqa: BLE001
        return f'<{type(e).__name__}: {e}>'


def section(name):
    print(f'\n===== {name} =====', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input')
    ap.add_argument('--output')
    a, unknown = ap.parse_known_args()

    section('python / os')
    print('python', sys.version.replace('\n', ' '))
    print('platform', platform.platform(), '| cpu', os.cpu_count())
    print('cwd', os.getcwd(), '| argv', sys.argv[1:], '| unknown', unknown)
    print('user', sh('whoami'), '| home', os.path.expanduser('~'))
    print('memory', sh('free -g | head -2'))

    section('gpu')
    print(sh('nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv'))

    section('packages')
    for pkg in ['vllm', 'torch', 'transformers', 'openai', 'pandas', 'pyarrow', 'numpy', 'sentence_transformers',
                'accelerate', 'flash_attn', 'xformers', 'sklearn', 'lightgbm', 'catboost', 'natasha', 'huggingface_hub']:
        try:
            m = __import__(pkg)
            print(f'{pkg:22s} {getattr(m, "__version__", "?")}')
        except Exception as e:  # noqa: BLE001
            print(f'{pkg:22s} нет ({type(e).__name__})')

    section('env (выборочно, без секретов)')
    for k in ['CUDA_VISIBLE_DEVICES', 'HF_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_ENDPOINT',
              'TRANSFORMERS_CACHE', 'VLLM_USE_V1', 'PIP_INDEX_URL']:
        print(k, '=', os.environ.get(k))
    print('http(s)_proxy задан:', any(os.environ.get(k) for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']))

    section('disk')
    for p in [os.getcwd(), os.path.expanduser('~'), '/tmp']:
        try:
            t, u, f = shutil.disk_usage(p)
            print(f'{p}: свободно {f / 2**30:.1f} ГБ из {t / 2**30:.1f} ГБ')
        except Exception as e:  # noqa: BLE001
            print(p, e)

    section('сеть')
    for url in ['https://huggingface.co', 'https://pypi.org/simple/pip/', 'https://modelscope.cn',
                'https://hf-mirror.com', 'https://github.com']:
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=6) as r:
                print(f'{url}: {r.status} за {time.time() - t0:.1f} с')
        except Exception as e:  # noqa: BLE001
            print(f'{url}: недоступен ({type(e).__name__}: {str(e)[:80]})')

    section('модели на машине')
    for root in [os.path.expanduser('~/.cache/huggingface'), os.path.expanduser('~/.cache/modelscope'),
                 '/models', '/model', '/data', '/mnt', '/opt', '/home/jovyan', '/workspace', '/shared']:
        if os.path.isdir(root):
            print(root, '->', sh(f'ls -la {root} | head -25'))
    print('safetensors (до 30, глубина 5):', sh('find / -maxdepth 5 -name "*.safetensors" 2>/dev/null | head -30', 60))

    section('входные данные')
    if a.input:
        p = Path(a.input)
        print('input', p, 'существует:', p.exists(), '| размер', p.stat().st_size if p.exists() else None)
        try:
            import pandas as pd
            df = pd.read_parquet(p) if str(p).endswith('.parquet') else pd.read_csv(p)
            print('строк', len(df), '| колонки', list(df.columns))
            if 'prompt' in df.columns:
                print('длина prompt, символов: медиана', int(df['prompt'].str.len().median()),
                      'максимум', int(df['prompt'].str.len().max()))
        except Exception as e:  # noqa: BLE001
            print('не прочитался:', e)
    print('output', a.output)

    section('конец диагностики')
    sys.exit(3)   # намеренная ошибка: платформа покажет этот лог, попытка вернётся


if __name__ == '__main__':
    main()
