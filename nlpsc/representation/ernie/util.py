# encoding:utf-8


def split_text(text, max_seq_len):
    texts = []
    for i in range(len(text) // max_seq_len + 1):
        sub = text[i * max_seq_len:(i + 1) * max_seq_len]
        texts.append(sub)
    return texts
