import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MyDataset(Data.Dataset):
    def __init__(self, path, train=True):
        # TODO
        # 1. Initialize file path or list of file names.
        self.root = './20newsgroups/3/'
        self.train = train

    def __getitem__(self, index):
        """
        if self.train is True:
            x = np.load(self.root + 'train_data_' + str(self.task) + '_' + str(index) + '.npy').reshape(1, -1)
        else:
            x = np.load(self.root + 'test_data_' + str(self.task) + '_' + str(index) + '.npy').reshape(1, -1)
        x = torch.FloatTensor(x)
        target = int(self.label[index])
        return x, target
        """
        pass

    def __len__(self):
            pass
            return self.size

if __name__ == '__main__':
    x = unpickle("./cifar-10-batches-py/data_batch_1")
    print(x[b"data"].shape)
    d = x[b"data"][0].reshape(3, 32, 32)