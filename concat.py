import pickle
import torch
import logging


from imp import reload
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
for batch in range(44972):
    file1 = 'processVec/Direct-LSTM-Oneway_' + str(batch) + '.pkl'
    file2 = 'processVec/Direct-LSTM-Oneway_Rev_' + str(batch) + '.pkl'
    tensor1 = pickle.load(open(file1, 'rb'))
    tensor2 = pickle.load(open(file2, 'rb'))
    length = len(tensor1)
    new_list = []
    for idx in range(length):
        new_tensor = torch.cat((tensor1[idx][0], tensor2[length - 1 - idx][0]), 1)
        new_list.append(new_tensor)
    file3 = 'processVec/BiLSTM_' + str(batch) + '.pkl'
    pickle.dump(new_list, open(file3, 'wb'))
    del new_list
    logging.info('* News: %d/%d Finished!' %(batch, 44972))