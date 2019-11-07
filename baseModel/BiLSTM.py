import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.tensor import Tensor
from torch import device as D
import pickle
from torch import FloatTensor as FT
from gensim.models.fasttext import FastText
import time
import logging


def runModel(coreTensor, inputSize: int, hiddenSize: int, d, model):
    '''
    This function runs LSTM for one-way
    :param coreTensor:
    :return:
    '''
    # d = D('cuda')
    # logging.info('Core Tensor dumping.......')
    # pickle.dump(coreTensor, open('coreTensor.pkl', 'wb'))
    # logging.info('Core Tensor dumped.')
    logging.info('Core Tensor loading.......')
    coreTensor = pickle.load(open('coreTensor.pkl', 'rb'))
    logging.info('Core Tensor loaded.')
    coreLSTM = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize).to(d)
    batch = 0
    logging.info('LSTM Start.(Core-One way)')
    retList = []
    sum_time = 0
    for news_tensor in coreTensor:
        time_start = time.time()
        inputs_2 = []
        for word in news_tensor:
            inputs_2.append(model[word])
        inputs = Tensor(inputs_2).to(d)
        hidden = (torch.rand(1, 1, hiddenSize).to(d), torch.rand(1, 1, hiddenSize).to(d))
        iter_time = 0
        out = Tensor().to(d)
        out_list = []
        for i in inputs:
            time_per_iter = time.time()
            out, hidden = coreLSTM(i.view(1, 1, -1), hidden)
            out_list = [out]

            # logging.info('\t\t + Iter: %d/%d Time: %.3lf s, remain %d iterations, ETA: %.3lf s' % (
            # iter_time + 1, len(inputs), time.time() - time_per_iter, len(inputs) - iter_time - 1,
            # (len(inputs) - iter_time - 1) * (time.time() - time_per_iter)))

            iter_time = iter_time + 1
        sum_time += int(time.time() - time_start)
        avg_time = 1.0 * sum_time / (batch + 1)
        remainSecTot = 1.0 * (len(coreTensor) - batch - 1) * (avg_time)
        remainMin = (int(remainSecTot) % 3600) // 60
        remainHour = int(remainSecTot) // 3600
        remainSec = int(remainSecTot) % 60
        logging.info('* News: %d/%d,Time: %.2lfs, remain %d news, ETA: %d:%d:%d s' % (
        batch + 1, len(coreTensor), time.time() - time_start, len(coreTensor) - batch - 1, remainHour, remainMin, remainSec))
        fileName = 'processVec/Direct-LSTM-Oneway_Sentence_' + str(batch) + '.pkl'
        foutput = open(fileName, 'wb')
        pickle.dump(out_list, foutput)
        # retList.append(out)
        batch = batch + 1
        del out
        del out_list
        del inputs_2
        del hidden
    logging.info('LSTM Finish.(Core-One way)')
    return retList


def biLSTM(all_corpus: list, model, d=None) -> (Tensor, object):
    if d is None:
        d = D('cpu')
    else:
        d = D('cuda')
    MatrixRet = []
    size = 0
    totalLength = len(all_corpus)
    cmtt = 0
    for topic in all_corpus:
        word_list_vec = []
        for news in topic:
            for sentence in news:
                word_list = sentence.split(' ')
                for word in word_list:
                    if word != '' and word != '\n' and word != ' ' and word != '  ':
                        vec = model[word]
                        word_list_vec.append(word)
                        size = len(vec)
            # word_list_vec = FT(word_list_vec)
            MatrixRet.append(word_list_vec)
        cmtt = cmtt + 1
        logging.info('\t * News: %d/%d' % (cmtt, totalLength))
    # for topic in all_corpus:
    #     word_list_vec = []
    #     for news in topic:
    #         if size != 0:
    #             break
    #         for sentence in news:
    #             if size != 0:
    #                 break
    #             word_list = sentence.split(' ')
    #             for word in word_list:
    #                 if word != '' and word != '\n' and word != ' ' and word != '  ':
    #                     vec = model[word]
    #                     word_list_vec.append(word)
    #                     size = len(vec)
    #                     break
    #     # word_list_vec = FT(word_list_vec)
    #     MatrixRet.append(word_list_vec)
    #     cmtt = cmtt + 1
    #     logging.info('\t * News: %d/%d' % (cmtt, totalLength))
    ret1st = runModel(MatrixRet, size, size, d, model)
    return (Tensor([0,0]), ret1st)
