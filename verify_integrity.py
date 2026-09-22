"""Read-only verification of the extension release; uses Python's standard library."""
import hashlib
import json
from pathlib import Path

SOURCE_DIRS = ('scripts', 'javascript', 'pi_ideogram_assets', 'pi_ideogram_forge',
               'pi_ideogram_lib', 'pi_ideogram_lora', 'docs', '.github', 'tests')
SOURCE_EXTENSIONS = {'.py', '.js', '.cjs', '.css', '.json', '.txt', '.md', '.yml', '.yaml'}


def source_files(root):
    files = []
    for folder in SOURCE_DIRS:
        for path in (root / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in SOURCE_EXTENSIONS:
                files.append(path)
    files.extend(p for p in root.iterdir() if p.is_file() and (
        p.suffix in SOURCE_EXTENSIONS or p.name in {'LICENSE', 'NOTICE', '.gitignore', '.gitattributes'})
        and p.name not in {'config.json', 'release-manifest.json'})
    return sorted(files)


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'release-manifest.json').read_text(encoding='utf-8'))
    errors = []
    expected = manifest['files']
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            errors.append(f'Unsafe manifest path: {name}')
        elif not path.is_file():
            errors.append(f'Missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f'Changed: {name}')
    actual = {p.relative_to(root).as_posix() for p in source_files(root)}
    errors.extend(f'Unexpected source: {name}' for name in sorted(actual - expected.keys()))
    return errors


if __name__ == '__main__':
    try:
        problems = verify(Path(__file__).resolve().parent)
    except (OSError, ValueError, KeyError) as error:
        print(f'Cannot verify this copy: {error}')
        raise SystemExit(2)
    print('\n'.join(problems) if problems else 'PASS: release source matches the recorded hashes.')
    raise SystemExit(bool(problems))
