"""Knowledge base: Markdown documents chunked by heading and ranked with Okapi BM25.

Pure Python, no embeddings: documents live in ``<data_dir>/knowledge/{es,pt,en}/*.md``;
each ``## heading`` becomes one passage. Tokens are case- and accent-folded, stopwords of
the passage's language are dropped and plurals are lightly stripped (``wa_assistant.text``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from wa_assistant.text import LANGUAGES, Lang, tokenize

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True, slots=True)
class Passage:
    doc_id: str  # "es/hours"
    lang: Lang
    title: str  # the document's H1, used for the source tag
    heading: str  # the section heading (the title for the text before the first section)
    body: str


@dataclass(frozen=True, slots=True)
class ScoredPassage:
    passage: Passage
    score: float


def chunk_markdown(text: str, doc_id: str, lang: Lang) -> list[Passage]:
    """Split one Markdown document into passages: one per heading below the title."""
    title = doc_id.rsplit("/", 1)[-1]
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            title = match.group(2)
        elif match:
            sections.append((match.group(2), []))
        else:
            sections[-1][1].append(line)
    passages = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            passages.append(Passage(doc_id, lang, title, heading or title, body))
    return passages


class BM25Index:
    """Okapi BM25 over a fixed list of passages (k1=1.5, b=0.75)."""

    def __init__(
        self, documents: Sequence[tuple[Passage, list[str]]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self._passages = [passage for passage, _ in documents]
        self._term_freqs = [Counter(tokens) for _, tokens in documents]
        self._lengths = [len(tokens) for _, tokens in documents]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._k1 = k1
        self._b = b
        doc_freq: Counter[str] = Counter()
        for freqs in self._term_freqs:
            doc_freq.update(freqs.keys())
        total = len(self._passages)
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, query_tokens: Sequence[str], k: int) -> list[ScoredPassage]:
        """Top ``k`` passages with a positive score, best first."""
        terms = [term for term in dict.fromkeys(query_tokens) if term in self._idf]
        if not terms:
            return []
        scored: list[ScoredPassage] = []
        for index, freqs in enumerate(self._term_freqs):
            norm = self._k1 * (1 - self._b + self._b * self._lengths[index] / self._avg_length)
            score = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (self._k1 + 1) / (tf + norm)
            if score > 0:
                scored.append(ScoredPassage(self._passages[index], score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


def _passage_tokens(passage: Passage, lang: Lang | None) -> list[str]:
    # Index the section heading with its text; repeating the document title in every
    # section would make all sections of a document look alike.
    return tokenize(f"{passage.heading}\n{passage.body}", lang)


class KnowledgeBase:
    """Per-language BM25 indexes plus one index over every language (the fallback)."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        self.passages = tuple(passages)
        self._by_lang = {
            lang: BM25Index([(p, _passage_tokens(p, lang)) for p in passages if p.lang == lang])
            for lang in LANGUAGES
        }
        self._all = BM25Index([(p, _passage_tokens(p, None)) for p in passages])

    @classmethod
    def load(cls, root: Path) -> KnowledgeBase:
        """Load every ``root/<lang>/*.md`` file (sorted, so the order is reproducible)."""
        passages: list[Passage] = []
        for lang in LANGUAGES:
            folder = root / lang
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.md")):
                text = path.read_text(encoding="utf-8")
                passages.extend(chunk_markdown(text, f"{lang}/{path.stem}", lang))
        if not passages:
            msg = f"no knowledge documents found under {root}"
            raise ValueError(msg)
        return cls(passages)

    def search(
        self, query: str, lang: Lang, k: int = 3, fallback_below: float = 1.5
    ) -> list[ScoredPassage]:
        """Top ``k`` passages in ``lang``; when those are missing or weak (best score below
        ``fallback_below``) the all-language index is tried and the stronger result wins."""
        local = self._by_lang[lang].search(tokenize(query, lang), k)
        if local and local[0].score >= fallback_below:
            return local
        everywhere = self._all.search(tokenize(query, None), k)
        if everywhere and (not local or everywhere[0].score > local[0].score):
            return everywhere
        return local
