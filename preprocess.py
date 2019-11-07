import pickle
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText

import logging
from baseModel import wordEmb


def _preprocess(name: str) -> list:
    train_data = []
    with open('data/multi-news-original/' + name, 'r') as f:
        train_data = f.readlines()
    totalLength = len(train_data)
    counter = 0
    all_corpus = []
    for i in range(len(train_data)):
        try:
            story = train_data[i].split('|||||')
            train_data[i] = []
            for j in story:
                if len(j) != 0:
                    train_data[i].append(j)
            logging.info('  - Current train_data size %d' % (len(train_data[i])))
            for j in range(len(train_data[i])):
                new_list = train_data[i][j].split(' NEWLINE_CHAR ');
                train_data[i][j] = []
                for q in new_list:
                    if len(q) != 0 and q != ' ' and q != '  ' and q != '\n':
                        # logging.info('q type:%s' %(type(q)))
                        train_data[i][j].append('<!START> ' + q + ' <!END>')
                        all_corpus.append(['<!START>'] + q.split(' ') + ['<!END>'])
            logging.info('Data %d/%d' % (counter, totalLength))
        except Exception as identifier:
            logging.error('Error in %d / %d, %s' % (counter, totalLength, identifier))
        counter = counter + 1
    return train_data, all_corpus


if __name__ == '__main__':
    from imp import reload

    reload(logging)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    train_source, train_all_corpus = preprocess('train.src')
    fasttext_model = wordEmb.getWordVec(train_all_corpus, 2)
    pickle.dump(fasttext_model, open('fasttext_model.pkl', 'wb'))

