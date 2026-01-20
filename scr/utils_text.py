#utils_text.py
#Arquivo suporte para manipulação de textos

from __future__ import annotations
import re
from typing import Dict, Iterable


def normalize_text_ptbr(text: str, allow_numbers: bool = True, allow_apostrophe: bool = True) -> str:
    text = (text or "").lower()

    if not allow_numbers:
        text = re.sub(r"\d+", "", text)

    if not allow_apostrophe:
        text = text.replace("'", "")

    # Vocabulario limitado a letras e espaço
    text = re.sub(r"[^a-zà-úçãõáéíóúâêîôûäëïöüñ ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab_from_texts(
    texts: Iterable[str],
    allow_numbers: bool = True,
    allow_apostrophe: bool = True,
) -> Dict[str, int]:
    """
    Retorna vocab no formato esperado pelo Wav2Vec2CTCTokenizer (json com char->id).
    Inclui tokens especiais e delimiter.
    """
    chars = set()
    for t in texts:
        t = normalize_text_ptbr(t, allow_numbers=allow_numbers, allow_apostrophe=allow_apostrophe)
        chars.update(set(t))

    # CTC: espaço vira delimiter "|"
    if " " in chars:
        chars.remove(" ")

    vocab_list = sorted(list(chars))
    vocab = {c: i for i, c in enumerate(vocab_list)}

    # Adiciona delimiter e especiais no final
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab
