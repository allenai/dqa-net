import re

def tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    tokens = [token.lower() for token in tokens]
    return tokens
