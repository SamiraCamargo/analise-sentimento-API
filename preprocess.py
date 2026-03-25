import re

def clean_text(text):
    # Converter para string
    text = str(text)

    # Deixar tudo em minúsculas
    text = text.lower()

    # Remover tags HTML como <br />
    text = re.sub(r"<.*?>", " ", text)

    # Remover espaços extras
    text = re.sub(r"\s+", " ", text).strip()

    return text