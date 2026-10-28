"""Verify released source/data hashes; generated figures are excluded."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def payload_files():
    return {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
            if p.is_file() and not any(x in p.relative_to(ROOT).parts
               for x in ('.git', '__pycache__', 'figures'))
            and p.name != 'manifest.json'}


def main():
    expected = {r['path']: r for r in json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))['files']}
    actual = payload_files()
    if set(actual) != set(expected):
        raise SystemExit(f'File list differs. Missing={sorted(set(expected)-set(actual))}; unlisted={sorted(set(actual)-set(expected))}')
    for name, p in actual.items():
        data = p.read_bytes()
        if len(data) != expected[name]['bytes'] or hashlib.sha256(data).hexdigest() != expected[name]['sha256']:
            raise SystemExit(f'Hash/size mismatch: {name}')
    print(f'Manifest verified: {len(actual)} source/data files.')


if __name__ == '__main__':
    main()
