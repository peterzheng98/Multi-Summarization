from baseModel import wordEmb, BiLSTM
import logging
import preprocess
from gensim.models import FastText
from gensim.models import Word2Vec
# import pandas as pd
import logging
import tensorflow as tf
import numpy as np
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec, Callback, CoherenceMetric, ConvergenceMetric, DiffMetric

def main():
    from imp import reload
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    train_source, train_all_corpus = preprocess._preprocess('train.src')
    # fasttext_model = wordEmb.getWordVec(train_all_corpus, 2)
    fasttext_model = FastText.load('FastText.model')
    _, singleDirection = BiLSTM.biLSTM(train_source, fasttext_model, '111')

