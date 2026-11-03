"""Check repository-local Markdown links and images, without network access.

Run from any directory: python scripts/check_repository_navigation.py
URL fragments are preserved but not validated; external URLs are not fetched.
"""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\s*\)")


def main():
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode().split("\0")
    documents = sorted({name for name in tracked if name.endswith(".md") and (ROOT/name).is_file()})
    checked, failures = 0, []
    for name in documents:
        path = ROOT/name
        fence = None
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                token = marker.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
                continue
            if fence:
                continue
            for match in INLINE_LINK.finditer(line):
                url = urlsplit(match.group(1).strip("<>"))
                if url.scheme or url.netloc or not url.path:
                    continue
                target = unquote(url.path)
                destination = ROOT/target.lstrip("/") if target.startswith("/") else path.parent/target
                checked += 1
                if not destination.exists():
                    failures.append(f"{name}:{line_number}: missing {target}")
    for failure in failures:
        print(failure)
    print(f"Checked {checked} local links/images in {len(documents)} Markdown files; {len(failures)} missing.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
