"""
Helper utilities for text cleaning, scoring, and UI formatting.
"""


def clean_text_string(text: str) -> str:
    """Normalize whitespace and remove invalid characters from raw text."""
    if not text:
        return ""
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    # Strip leading/trailing whitespace from lines and collapse consecutive spacing
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join([line for line in lines if line])

    # Remove common header/footer patterns like "Page X" or "Page X of Y"
    import re

    text = re.sub(r"(?i)\n\s*page\s+\d+\s*(?:of\s*\d+)?\s*\n", "\n", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_confidence_score(raw_score: float) -> str:
    """Format similarity or reranker score into a clean percentage string."""
    # Convert arbitrary raw score/logit to bounded 0-100% representation
    if raw_score <= 1.0:
        score_pct = max(0.0, min(100.0, raw_score * 100.0))
    elif raw_score <= 10.0:
        # e.g., 1.5 -> 100% when input is logit > 1.0 or similarity > 1.0
        score_pct = min(
            100.0,
            (
                raw_score * 100.0
                if raw_score <= 1.0
                else (raw_score if raw_score > 2.0 else 100.0)
            ),
        )
    else:
        score_pct = min(100.0, raw_score)
    return f"{score_pct:.1f}%"


def truncate_snippet(text: str, max_length: int = 180) -> str:
    """Truncate text for card or preview display."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."
