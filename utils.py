import re


def tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    tokens = [token.lower() for token in tokens]
    return tokens


def vget(vocab_dict, word):
    return vocab_dict[word] if word in vocab_dict else 0

def vlup(vocab_dict, words):
    return [vget(vocab_dict, word) for word in words]
