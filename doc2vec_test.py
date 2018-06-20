import logging
from gensim.models.doc2vec import (
    Doc2Vec,
    TaggedDocument,
)

logging.basicConfig(
    format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
    level=logging.DEBUG,
)


def to_str(d):
    return ", ".join(d.keys())


SENTS = [
    "anecdotal using a personal experience or an isolated example instead of a sound argument or compelling evidence",
    "plausible thinking that just because something is plausible means that it is true",
    "occam razor is used as a heuristic technique discovery tool to guide scientists in the development of theoretical models rather than as an arbiter between published models",
    "karl popper argues that a preference for simple theories need not appeal to practical or aesthetic considerations",
    "the successful prediction of a stock future price could yield significant profit",
]

SENTS = [s.split() for s in SENTS]


def main():
    sentences_1 = [
        TaggedDocument(SENTS[0], tags=['SENT_0']),
        TaggedDocument(SENTS[1], tags=['SENT_1']),
        TaggedDocument(SENTS[2], tags=['SENT_2']),
    ]
    sentences_2 = [
        TaggedDocument(SENTS[3], tags=['SENT_3']),
        TaggedDocument(SENTS[4], tags=['SENT_4']),
    ]

    model = Doc2Vec(min_count=1, workers=4)

    model.build_vocab(sentences_1)
    model.train(sentences_1, total_examples=model.corpus_count, epochs=model.iter)

    print("-- Base model")
    print("Vocabulary:", to_str(model.wv.vocab))
    print("Tags:", to_str(model.docvecs.doctags))

    print("Updating model....")
    model.build_vocab(sentences_2, update=True)
    model.train(sentences_2, total_examples=model.corpus_count, epochs=model.iter)

    print("-- Updated model")
    print("Vocabulary:", to_str(model.wv.vocab))
    print("Tags:", to_str(model.docvecs.doctags))

    token_list = "the successful prediction of a stock future price could yield significant profit".split()
    infer_vector = model.infer_vector(token_list)
    # print(model.docvecs.most_similar(positive=[infer_vector]))

if __name__ == '__main__':
    main()