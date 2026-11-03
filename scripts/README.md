# Repository maintenance

[Project home](../README.md) · [Contributing](../CONTRIBUTING.md)

Run `python scripts/check_repository_navigation.py` to find broken repository-local Markdown links and image paths. It checks tracked files and new nonignored Markdown files, skips fenced code examples, and runs without network access. It does not validate URL fragments or external websites.

Scientific reproduction commands live in [benchmarks/](../benchmarks/README.md); numerical implementation lives in [src/](../src/README.md).
