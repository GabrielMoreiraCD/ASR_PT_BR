#Arquivo suporte para manipulação de textos

def normalize_text_ptbr(text: str, allow_numbers: bool = True, allow_apostrophe: bool = True) -> str:
    import re
    text = text.lower()
    if not allow_numbers:
        text = re.sub(r'\d+', '', text)
    if not allow_apostrophe:
        text = text.replace("'", "")
    # Remove outros caracteres indesejados
    text = re.sub(r'[^a-zà-úçãõáéíóúâêîôûäëïöüñ ]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab_from_texts(texts: list[str]) -> set[str]:
    vocab = set()
    for text in texts:
        vocab.update(set(text))
    return vocab