"""Shared NLTK utility to avoid duplicating _safe_nltk_download across modules."""
import nltk


def safe_nltk_download(pkg):
    """Download an NLTK resource if not already present, suppressing errors."""
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg.split('/')[-1], quiet=True, raise_on_error=False)
        except Exception:
            pass
