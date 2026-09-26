"""Duplicate and junk detection for OpenAlex works.

OpenAlex often holds the same paper under several IDs: Zenodo/figshare versions, preprint + journal version,
translations. In the demo corpus (2026-09-26) that was ~12% of all works, and one paper took two of the top 5 slots
after reranking. Two works count as duplicates if their normalized titles *or* their normalized abstracts are equal:
versions often differ in only one of the two.
"""

import hashlib
import re
import unicodedata
from collections.abc import Iterable

# Shorter "abstracts" are citation stubs ("Current BiologyDOI: https://doi.org/... ISSN"), file names ("som_R2_v2.docx")
# or placeholders ("International audience"). Their embeddings are meaningless and the reranker can't judge them.
MIN_ABSTRACT_WORDS = 20


def text_key(text: str | None) -> str | None:
    """Hash of the text ignoring case, punctuation and whitespace; None if nothing is left to compare."""
    if not text:
        return None
    normalized = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
    if not normalized:
        return None
    return hashlib.sha1(normalized.encode()).hexdigest()


def has_usable_abstract(abstract: str | None) -> bool:
    return bool(abstract) and len(abstract.split()) >= MIN_ABSTRACT_WORDS


class DuplicateFilter:
    """Remembers the title and abstract keys of the works kept so far."""

    def __init__(self, title_keys: Iterable[str] = (), abstract_keys: Iterable[str] = ()):
        self.title_keys = set(title_keys)
        self.abstract_keys = set(abstract_keys)

    def is_new(self, title: str | None, abstract: str | None) -> bool:
        """Returns False for a duplicate of a kept work; otherwise keeps this work and returns True."""
        title_key, abstract_key = text_key(title), text_key(abstract)
        if title_key in self.title_keys or abstract_key in self.abstract_keys:
            return False
        if title_key:
            self.title_keys.add(title_key)
        if abstract_key:
            self.abstract_keys.add(abstract_key)
        return True
