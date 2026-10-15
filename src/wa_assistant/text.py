"""Text normalisation shared by retrieval, language detection and intent matching."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, get_args

Lang = Literal["es", "pt", "en"]
LANGUAGES: tuple[Lang, ...] = get_args(Lang)

_WORD_RE = re.compile(r"[^\W_]+")


def fold(text: str) -> str:
    """Case-fold and strip accents: ``"Mañana"`` -> ``"manana"``, ``"Atenção"`` -> ``"atencao"``."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def words(text: str) -> list[str]:
    """Folded word tokens (letters and digits only)."""
    return _WORD_RE.findall(fold(text))


def stem(word: str) -> str:
    """Very light plural stripping, applied identically to documents and queries.

    ``"precios" -> "precio"``, ``"sociales" -> "social"``, ``"cartoes" -> "cartao"``,
    ``"prices" -> "price"``. It only has to be consistent, not linguistically perfect.
    """
    if len(word) > 4 and word.endswith("oes"):
        return word[:-3] + "ao"
    if len(word) > 4 and word.endswith("es") and word[-3] in "lrndz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stopwords(raw: str) -> frozenset[str]:
    return frozenset(raw.split())


# Small, folded stopword lists. Interrogatives that carry meaning for retrieval
# ("donde", "cuanto", "quanto", "much") are deliberately NOT stopwords.
STOPWORDS: dict[Lang, frozenset[str]] = {
    "es": _stopwords(
        "a al algo algun alguna como con cual cuales de del el ella en es esa ese eso esta este"
        " esto estoy hay la las le les lo los me mi mis muy nos o para pero por porque que se"
        " si sin son su sus te tu tus un una uno unos unas y ya yo hola gracias favor quiero"
        " queria quisiera necesito puedo podria usted ustedes vos tengo hace hacen tiene tienen"
    ),
    "pt": _stopwords(
        "a ao aos as com como da das de do dos e ela ele em essa esse isso esta este eu me meu"
        " minha na nas no nos o os ou para pela pelo por que se sem seu sua um uma uns umas"
        " voce voces oi ola obrigado obrigada quero queria gostaria preciso posso favor faz fazem"
        " tem"
    ),
    "en": _stopwords(
        "a an and are at be can could do does for from have hi hello how i im in is it me my of"
        " on or our please the there to us we what when where which who will with would you"
        " your thanks thank want need"
    ),
}
ALL_STOPWORDS: frozenset[str] = frozenset().union(*STOPWORDS.values())


def tokenize(text: str, lang: Lang | None = None) -> list[str]:
    """Retrieval tokens: folded, stopwords of ``lang`` removed (all languages when ``None``), stemmed."""
    stop = STOPWORDS[lang] if lang is not None else ALL_STOPWORDS
    return [
        stem(word) for word in words(text) if word not in stop and (len(word) > 1 or word.isdigit())
    ]
