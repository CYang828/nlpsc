import plac
from pathlib import Path
# import jieba_fast as jieba
import jieba


def cut_words(fin, fout, userdict, stopword):
    stopwords = []
    if userdict:
        jieba.load_userdict(userdict)
    if stopword:
        stopwords = [line.strip() for line in open(stopword, 'r', encoding='utf-8').readlines()]

    count = 0
    with open(fin, mode='r', encoding='utf-8') as infile, open(fout, mode='w', encoding='utf-8') as outfile:
        for lines in infile:
            line = lines.strip()
            if line.startswith('doc'):
                if line == 'doc':
                    outfile.write('\n')
                    count = count + 1
                    if count % 1000 == 0:
                        print('%s articles were finished.......' % count)
                continue
            for word in jieba.cut(line):
                if word not in stopwords:
                    outfile.write(word + ' ')
    print('%s articles were finished.......' % count)


@plac.annotations(
    input_loc=("Input file location", "positional", None, Path),
    output_loc=("Input file location", "positional", None, Path),
    userdict_loc=("Userdict location", "option", "u", Path),
    stopword_loc=("stopword location", "option", "s", Path),
)
def main(input_loc, output_loc, userdict_loc, stopword_loc):
    cut_words(input_loc, output_loc, userdict_loc, stopword_loc)


if __name__ == '__main__':
    plac.call(main)

