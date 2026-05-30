"""
pymrsf.chunker — Surprise-guided auto-chunking

Instead of splitting text at fixed sizes or sentence boundaries, smart_chunk()
uses the model's own surprise signal to find natural knowledge boundaries.

How it works:
  1. Run get_surprises(text) — the model scores every token for "surprise"
  2. Compute a rolling average of surprise over a sliding window
  3. Find positions where the rolling average drops sharply after a local peak
     → these are "absorption points" where the model finished learning a concept
  4. Map token positions back to character offsets and split there
  5. Merge chunks that are too short; split chunks that are too long

This produces chunks that are semantically coherent from the model's perspective,
unlike arbitrary character/sentence splitting.

Requires local provider (get_surprises uses raw logits).
Falls back to sentence-based splitting for API providers.
"""

import logging
import re

from .core import detokenize, get_surprises, provider_capabilities, tokenize

_logger = logging.getLogger("pymrsf.chunker")


def _sentence_fallback(text: str, max_chunk_len: int) -> list[str]:
    """Simple sentence-based chunking used as fallback for API providers."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > max_chunk_len and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def smart_chunk(
    text: str,
    min_chunk_len: int = 100,
    max_chunk_len: int = 1000,
    surprise_drop_threshold: float = 0.4,
    window: int = 5,
) -> list[str]:
    """Split text into semantically coherent chunks using the model's surprise signal.

    Chunk boundaries are placed where the rolling-average surprise drops
    significantly after a local peak — indicating the model has absorbed a
    new knowledge unit and "settled" before the next concept begins.

    Requires local provider. Falls back to sentence splitting for API providers.

    Args:
        text                   : Document text to chunk
        min_chunk_len          : Minimum characters per chunk (merge shorter ones)
        max_chunk_len          : Maximum characters per chunk (force-split longer ones)
        surprise_drop_threshold: Fractional drop in rolling surprise to trigger a boundary
                                 (0.4 = 40% drop from local peak triggers a cut)
        window                 : Rolling average window in tokens

    Returns:
        List of text chunk strings

    Example:
        from pymrsf import smart_chunk
        chunks = smart_chunk(long_article, min_chunk_len=200, max_chunk_len=800)
    """
    caps = provider_capabilities()
    if not caps.get("supports_logits", False):
        _logger.warning("Local provider not available — falling back to sentence chunking.")
        return _sentence_fallback(text, max_chunk_len)

    try:
        surprises_data = get_surprises(text)
    except Exception as e:
        _logger.warning("get_surprises failed (%s) — falling back to sentence chunking.", e)
        return _sentence_fallback(text, max_chunk_len)

    # surprises_data is expected to be a list of dicts: [{position, token_str, surprised}, ...]
    # or a list of (position, token_str) tuples — handle both formats
    if not surprises_data:
        return _sentence_fallback(text, max_chunk_len)

    # Normalise to list of (position, token_str, is_surprised)
    def _parse(item):
        if isinstance(item, dict):
            return item.get("position", 0), item.get("token", ""), bool(item.get("surprised", False))
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return item[0], item[1], True
        return 0, "", False

    parsed = [_parse(s) for s in surprises_data]

    # Build binary surprise signal (1=surprised, 0=not) indexed by token position
    max_pos = max(p for p, _, _ in parsed) + 1 if parsed else 1
    signal = [0.0] * max_pos
    for pos, _, is_surprised in parsed:
        if 0 <= pos < max_pos:
            signal[pos] = 1.0 if is_surprised else 0.0

    # Rolling average
    rolling = []
    for i in range(max_pos):
        start = max(0, i - window + 1)
        rolling.append(sum(signal[start:i + 1]) / (i - start + 1))

    # Find boundary positions: rolling drops by >= surprise_drop_threshold from local peak
    boundary_positions: list[int] = []
    peak = rolling[0] if rolling else 0.0
    for i in range(1, len(rolling)):
        if rolling[i] > peak:
            peak = rolling[i]
        elif peak > 0 and (peak - rolling[i]) / peak >= surprise_drop_threshold:
            boundary_positions.append(i)
            peak = rolling[i]  # reset peak after boundary

    # Map token positions to character offsets
    # Reconstruct token-by-token to get character positions
    try:
        token_ids = tokenize(text)
    except Exception:
        return _sentence_fallback(text, max_chunk_len)

    # Build cumulative char offsets in O(n) by detokenizing one token at a time.
    # boundary_positions are sorted ascending, so we walk token_ids once.
    set(boundary_positions)
    cum_len = 0
    cum_chars: list[int] = [0]  # cum_chars[i] = char offset just before token i
    for tid in token_ids:
        try:
            cum_len += len(detokenize([tid]))
        except Exception:
            pass
        cum_chars.append(cum_len)

    char_boundaries: list[int] = []
    for bp in boundary_positions:
        if bp < len(cum_chars):
            char_boundaries.append(cum_chars[bp])

    if not char_boundaries:
        return _sentence_fallback(text, max_chunk_len)

    # Split text at char boundaries
    raw_chunks: list[str] = []
    prev = 0
    for cb in sorted(set(char_boundaries)):
        raw_chunks.append(text[prev:cb])
        prev = cb
    raw_chunks.append(text[prev:])
    raw_chunks = [c for c in raw_chunks if c.strip()]

    # Post-process: merge short chunks, force-split long ones
    chunks: list[str] = []
    buffer = ""
    for chunk in raw_chunks:
        if len(buffer) + len(chunk) < min_chunk_len:
            buffer += (" " if buffer else "") + chunk.strip()
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = chunk.strip()
    if buffer:
        chunks.append(buffer.strip())

    # Force-split chunks that exceed max_chunk_len at sentence boundaries
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_len:
            final.append(chunk)
        else:
            final.extend(_sentence_fallback(chunk, max_chunk_len))

    return [c for c in final if c.strip()] or [text]
