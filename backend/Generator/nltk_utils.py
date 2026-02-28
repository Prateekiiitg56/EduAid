"""Shared NLTK utility to avoid duplicating _safe_nltk_download across modules."""
import logging
import nltk

logger = logging.getLogger(__name__)


def safe_nltk_download(pkg):
    """Download an NLTK resource if not already present, logging failures."""
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg.split('/')[-1], quiet=True, raise_on_error=False)
        except Exception as e:
            logger.warning("Failed to download NLTK resource '%s': %s", pkg, e)
