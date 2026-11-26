"""Verify that the original code files from the supplied archive are unchanged."""

import hashlib, json, logging
from pathlib import Path

try:
    from ._timing import LOG, configure_logging, step, progress
except ImportError:  # allow direct script invocation
    from _timing import LOG, configure_logging, step, progress


def main():
    configure_logging()
    try:
        with step("Original code verification", level=logging.INFO):
            result = _verify()
        if result:
            LOG.error("Original code verification found differences")
        return result
    except BaseException:
        LOG.exception("Original code verification could not complete")
        raise


def _verify():
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "original_code_sha256.json").read_text())
    root = directory.parent
    bad = []
    for name, h in progress(manifest.items(), "original file verification"):
        LOG.debug("Checking original file %s", name)
        p = root / name
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        if actual != h:
            bad.append(name)
            print("DIFFERENT OR MISSING:", name)
    if bad:
        print(
            "Original files differ from uploaded archive. They were not modified by this add-on; check compatibility."
        )
        return 1
    print(f"OK: {len(manifest)} original code/config files match the uploaded archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
