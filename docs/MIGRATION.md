# Migration Guide: v0.4 → v0.5+

If you built on pymrsf v0.4, this guide covers everything you need to know
to upgrade to v0.5.x. Most users need **zero code changes** — the breaking
changes are guarded by deprecation warnings and backward-compatible re-exports.

## Summary of changes

| Change | Impact | Action required |
|--------|--------|-----------------|
| Storage/inspect/benchmark → `pymrsf.experimental` | Import path change | Update imports (optional — re-exports still work) |
| `rebuild_faiss_from_sqlite()` deprecated | Calls still work but warn | Use `reset_index_metadata()` instead |
| Embed fail-fast by default | New behavior | Set `PYMRSF_ALLOW_PROVIDER_FALLBACK=true` to restore old behavior |
| `configure_logging()` added | No change | Call it to see library log output |
| `pymrsf.configure()` added | No change | Replaces direct env var hacks at runtime |
| `print()` → logging | No change | `import pymrsf` is now silent (opt-in logging) |
| Minimum Python | — | 3.9+ |

## 1. Filesystem moved to `pymrsf.experimental`

The MRSF delta-compression storage, inspection, and benchmarking modules
moved from `pymrsf.storage`, `pymrsf.inspect`, and `pymrsf.benchmark` into
`pymrsf.experimental`. The top-level re-exports (`from pymrsf import mrsf_write`,
`from pymrsf import mrsf_inspect`, etc.) **still work** for now but will be
removed in v0.6.

**Before (v0.4):**
```python
from pymrsf.storage import mrsf_write, mrsf_read
from pymrsf.inspect import mrsf_inspect
```

**After (v0.5+, recommended):**
```python
from pymrsf.experimental import mrsf_write, mrsf_read
from pymrsf.experimental import mrsf_inspect
```

**Still works but warns at import:**
```python
from pymrsf import mrsf_write  # OK, will be removed in v0.6
```

## 2. `rebuild_faiss_from_sqlite()` deprecated

This function was renamed to `reset_index_metadata()` to better reflect what
it does (rebuilds metadata, not embeddings). Calls to the old name emit a
`DeprecationWarning`.

**Before (v0.4):**
```python
from pymrsf import rebuild_faiss_from_sqlite
rebuild_faiss_from_sqlite()
```

**After (v0.5+):**
```python
from pymrsf import reset_index_metadata
reset_index_metadata()
```

## 3. Embed fail-fast is now default

In v0.4, if the embedding provider (Ollama) failed, pymrsf silently fell back
to an alternative provider. This masked configuration errors. In v0.5+, embed
failures **raise `RuntimeError` immediately** unless you opt in to fallback.

**To restore the v0.4 silent-fallback behavior:**
```bash
export PYMRSF_ALLOW_PROVIDER_FALLBACK=true
```

Or in code:
```python
import os
os.environ["PYMRSF_ALLOW_PROVIDER_FALLBACK"] = "true"
```

## 4. `configure_logging()` added

pymrsf now ships with a `logging.NullHandler` — importing the library produces
no console output unless you opt in. This replaces the old `print()`-based log
output that happened automatically on import.

**Before (v0.4):**
```python
import pymrsf  # printed model loading messages automatically
```

**After (v0.5+):**
```python
import pymrsf
pymrsf.configure_logging("INFO")        # see library messages
pymrsf.configure_logging("DEBUG")       # verbose debug output
pymrsf.configure_logging("WARNING")     # warnings and errors only
```

## 5. `pymrsf.configure()` for runtime settings

New `Config` dataclass and `pymrsf.configure()` function for changing settings
at runtime without environment variables.

```python
import pymrsf

pymrsf.configure(
    provider="openai",
    embed_timeout=60,
    default_relevance_cutoff=0.4,
)
```

## Minimum Python version

pymrsf v0.5+ requires **Python 3.9 or later**. Python 3.8 reached end-of-life
and is no longer tested.

## Quick checklist

- [ ] Search for `from pymrsf.storage` → change to `from pymrsf.experimental`
- [ ] Search for `rebuild_faiss_from_sqlite` → rename to `reset_index_metadata`
- [ ] Decide if you want embed fallback → set `PYMRSF_ALLOW_PROVIDER_FALLBACK=true`
- [ ] Consider calling `pymrsf.configure_logging("INFO")` in your entry point
