from gensim.models import FastText
from gensim.models import Word2Vec
import pandas as pd
import logging
import tensorflow as tf
import numpy as np
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec, Callback, CoherenceMetric, ConvergenceMetric, DiffMetric


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        logging.info("Epoch: %d / %d" % (self.epoch, model.epochs))

    def on_epoch_end(self, model):
        self.epoch = self.epoch + 1


epochLogger = EpochLogger()


def getWordVec(corpus: list, type=1) -> object:
    '''
        Args:
            corpus: list[list[str]], each sublist indicates a sentence
            type: 1 = word2vec, 2 = fasttext
    '''
    coherenceMetric = Callback(ConvergenceMetric())
    convergenceMetric = Callback(CoherenceMetric())
    diffMetric = Callback(DiffMetric())
    if type == 1:
        model = Word2Vec(corpus)
        model.save('word2vec.model')
        return model
    else:
        model = FastText(min_count=1)
        logging.info('Starting building vocabulary table')
        model.build_vocab(corpus)
        logging.info('Starting training')

        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[epochLogger])
        model.save('FastText.model')
        return model



