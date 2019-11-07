import torch
import torch.tensor as Tensor
from torch.autograd import Variable


def reverseSequence(inputs: Tensor, lengths: list[int], batch_first=True) -> Tensor:
    if batch_first:
        inputs = inputs.transpose(0, 1)
    maxLen, batchSize = inputs.size(0), inputs.size(1)

    if len(lengths) != batchSize:
        raise RuntimeError("inputs error with batchSize[%d] Required length[%d]" % (len(lengths), batchSize))

    ind = [list(reversed(range(0, l))) + list(range(l, maxLen)) for l in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = Variable(ind.expand_as(inputs))
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())

    reversedInput = torch.gather(inputs, 0, ind)
    if batch_first:
        reversedInput = reversedInput.transpose(0, 1)
    return reversedInput


def sort_batch(data, label, length):
    batch_size = data.size(0)
    inx = torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data = data[inx]
    label = label[inx]
    length = length[inx]
    length = list(length.numpy())
    return data, label, length
