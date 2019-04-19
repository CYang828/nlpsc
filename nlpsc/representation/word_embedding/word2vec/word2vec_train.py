import plac
import gensim
import multiprocessing
from pathlib import Path


def word2vec_train(infile, outfile, fmtfile, epoch, size, mini):
    sentences = gensim.models.word2vec.LineSentence(infile)
    model = gensim.models.Word2Vec(sentences,
                                   size=size,
                                   min_count=mini,
                                   sg=1,
                                   workers=multiprocessing.cpu_count(),
                                   iter=epoch)
    model.save(outfile)
    model.wv.save_word2vec_format(fmtfile, binary=False)
    # model.save_word2vec_format(output_file + '.vector', binary=True)


@plac.annotations(
    input_loc=("Location of input file", "positional", None, Path),
    output_model_loc=("Location of output model dump file", "positional", None, Path),
    output_format_loc=("Location of output format file", "positional", None, Path),
    epoch=("Train epoch", "option", "epoch", int),
    embedding_size=("Embedding size", "option", "size", int),
    min_count=("Corpus minimum size", "option", "min", int)
)
def main(input_loc, output_model_loc, output_format_loc, epoch=10, embedding_size=100, min_count=5):
    word2vec_train(input_loc, output_model_loc, output_format_loc, epoch, embedding_size, min_count)


if __name__ == "__main__":
    plac.call(main)