# Report digest validation (2026-11-16)

The validator at 678ab161c4047ac96d894891bb08e0b4aecc7ed4 claims to check SHA-256 hex digests but checks only string length. Identical 64-character non-hex placeholders can therefore pass the zero-failure equality check and yield `ok: true`.

The existing producer `_sha256_lines` in `src/jevguard_nsfa/benchmark_jev.py` uses `hashlib.sha256(...).hexdigest()`. The upstream search targeted Python SHA-256 hex validation. Inspected CPython 3.12 `Lib/string.py`, which defines `hexdigits` as ASCII digits plus a-f and A-F (https://github.com/python/cpython/blob/3.12/Lib/string.py; branch inspected 2026-11-16). CPython is PSF-licensed; this change directly uses its standard-library constant, not copied implementation. Python's hashlib contract describes hexadecimal output: https://docs.python.org/3.12/library/hashlib.html.

Use existing length/type checks plus membership in `string.hexdigits`. No new dependency, regex, numeric coercion, whitespace stripping or digest normalization. Preserve exact-string equality and accept both ASCII hex cases as before. This validates representation only, not authenticity or recomputation against original samples.

Verify each digest independently, equal malformed digests through the real CLI, valid uppercase/lowercase/zero digests, differing valid fingerprints, and mixed valid/invalid report batches. Run the entire offline suite and Ruff. No model or paid benchmark execution.
