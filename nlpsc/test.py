from nlpsc.shortcut import NLPShortcut


with NLPShortcut() as ns:
    corpus = ns.load_data_from_file('doc2text-output/')\
        .iter_clean()
        # .iter_stopword()
