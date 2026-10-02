"""Общие текстовые утилиты (токенизация, пересечение ключевых слов, фразы)."""

from __future__ import annotations

import re
from typing import List, Set

STOP_WORDS: Set[str] = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее кажется сейчас были куда зачем сказать всех".split()
)

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Разбить текст на значимые токены (без стоп-слов, нижний регистр)."""
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOP_WORDS and len(w) > 1]


def query_keyword_overlap(query: str, text: str) -> float:
    """Доля ключевых слов запроса, встречающихся в тексте чанка."""
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q:
        return 0.0
    return len(q & t) / len(q)


def phrase_match(query: str, text: str) -> float:
    """Доля биграмм запроса, встречающихся в тексте (точное вхождение фраз)."""
    qt = tokenize(query)
    if len(qt) < 2:
        return 0.0
    text_lower = text.lower()
    bigrams = [" ".join(qt[i : i + 2]) for i in range(len(qt) - 1)]
    hits = sum(1 for bg in bigrams if bg in text_lower)
    return hits / len(bigrams)
