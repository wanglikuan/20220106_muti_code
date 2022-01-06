# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import warnings
import numpy as np
import torch
import codecs
import scipy.misc as m
# import torchvision

class MNIST(data.Dataset):
# class MNIST(torchvision.datasets.MNIST):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, multi=False):
        # super().__init__(root, train=True, transform=None, target_transform=None, download=False)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.multi = multi

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if not self._check_multi_exists():
            raise RuntimeError('Multi Task extension not found.' +
                               ' You can use download=True to download it')

        self.data, self.target = self._load_data()
        if multi:
            if self.train:
                self.train_data, self.original_train_labels, self.train_labels\
                    = torch.load(os.path.join(self.root, self.processed_folder, self.multi_training_file))
            else:
                self.test_data, self.original_test_labels, self.test_labels\
                    = torch.load(os.path.join(self.root, self.processed_folder, self.multi_test_file))
        else:
            if self.train:
                self.train_data, self.train_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.training_file))
            else:
                self.test_data, self.test_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        import matplotlib.pyplot as plt
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.multi:
            if self.train:
                img, o_target, [target_0, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9] \
                    = self.train_data[index], \
                      self.original_train_labels, \
                      [self.train_labels[0][index],
                      self.train_labels[1][index],
                      self.train_labels[2][index],
                      self.train_labels[3][index],
                      self.train_labels[4][index],
                      self.train_labels[5][index],
                      self.train_labels[6][index],
                      self.train_labels[7][index],
                      self.train_labels[8][index],
                      self.train_labels[9][index]]
            else:
                img, o_target, [target_0, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9] \
                    = self.test_data[index], \
                      self.original_test_labels,\
                      [self.test_labels[0][index],
                      self.test_labels[1][index],
                      self.test_labels[2][index],
                      self.test_labels[3][index],
                      self.test_labels[4][index],
                      self.test_labels[5][index],
                      self.test_labels[6][index],
                      self.test_labels[7][index],
                      self.test_labels[8][index],
                      self.test_labels[9][index]]
                # img, [target_0, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9] \
                #     = self.test_data[index], \
                #       [self.test_labels_0[index],
                #       self.test_labels_1[index],
                #       self.test_labels_2[index],
                #       self.test_labels_3[index],
                #       self.test_labels_4[index],
                #       self.test_labels_5[index],
                #       self.test_labels_6[index],
                #       self.test_labels_7[index],
                #       self.test_labels_8[index],
                #       self.test_labels_9[index]]
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.multi:
            return img, o_target, [target_0, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9]
        else:
            return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _check_multi_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file))

    def _load_data(self):
        if self.train:
            data = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
            target = read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))

        else:
            # tmnist_ims, tmulti_mnist_ims, textension = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
            data = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
            target = read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
            # print(tmulti_mnist_labels_0)
            # print(len(tmulti_mnist_labels_0), tmulti_mnist_labels_0[0])

        return data, target


    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists() and self._check_multi_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        # mnist_ims, multi_mnist_ims, extension = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        mnist_ims = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        mnist_labels, multi_mnist_labels = read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))

        # tmnist_ims, tmulti_mnist_ims, textension = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
        tmnist_ims = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
        tmnist_labels, tmulti_mnist_labels = read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        # print(tmulti_mnist_labels_0)
        # print(len(tmulti_mnist_labels_0), tmulti_mnist_labels_0[0])

        mnist_training_set = (mnist_ims, mnist_labels)
        # multi_mnist_training_set = (multi_mnist_ims, multi_mnist_labels_l, multi_mnist_labels_r)
        multi_mnist_training_set = (mnist_ims, mnist_labels, multi_mnist_labels)

        mnist_test_set = (tmnist_ims, tmnist_labels)
        multi_mnist_test_set = (tmnist_ims, tmnist_labels, tmulti_mnist_labels)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(mnist_test_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


# re-write
def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        # multi_labels_l = np.zeros((1*length),dtype=np.long)
        # multi_labels_r = np.zeros((1*length),dtype=np.long)
        # for class_idx in range(10):
        multi_label_0 = np.zeros((1 * length), dtype=np.long)
        multi_label_1 = np.zeros((1 * length), dtype=np.long)
        multi_label_2 = np.zeros((1 * length), dtype=np.long)
        multi_label_3 = np.zeros((1 * length), dtype=np.long)
        multi_label_4 = np.zeros((1 * length), dtype=np.long)
        multi_label_5 = np.zeros((1 * length), dtype=np.long)
        multi_label_6 = np.zeros((1 * length), dtype=np.long)
        multi_label_7 = np.zeros((1 * length), dtype=np.long)
        multi_label_8 = np.zeros((1 * length), dtype=np.long)
        multi_label_9 = np.zeros((1 * length), dtype=np.long)
        #     locals()['multi_label_' + str(class_idx)] = np.zeros((1 * length), dtype=np.long)
        # print('multi_label_'+ str(i))
        for im_id in range(length):
            for class_idx in range(10):
                if class_idx == parsed[im_id]:
                    locals()['multi_label_' + str(class_idx)][im_id] = 1
                else:
                    locals()['multi_label_' + str(class_idx)][im_id] = 0
        return torch.from_numpy(parsed).view(length).long(), \
               [torch.from_numpy(multi_label_0).view(length * 1).long(),
               torch.from_numpy(multi_label_1).view(length * 1).long(),
               torch.from_numpy(multi_label_2).view(length * 1).long(),
               torch.from_numpy(multi_label_3).view(length * 1).long(),
               torch.from_numpy(multi_label_4).view(length * 1).long(),
               torch.from_numpy(multi_label_5).view(length * 1).long(),
               torch.from_numpy(multi_label_6).view(length * 1).long(),
               torch.from_numpy(multi_label_7).view(length * 1).long(),
               torch.from_numpy(multi_label_8).view(length * 1).long(),
               torch.from_numpy(multi_label_9).view(length * 1).long()]


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        # # parsed_label = np.frombuffer(data, dtype=np.uint8, offset=8)
        # # for i in range(10):
        # # # multi_label_0 = np.zeros((1*10), dtype=np.long)
        # #     locals()['multi_label_'+ str(i)] = np.zeros((1 * 10), dtype=np.long)
        # #     # print('multi_label_'+ str(i))
        # pv = parsed.reshape(length, num_rows, num_cols)
        # multi_length = length * 1
        # multi_data = np.zeros((1*length, num_rows, num_cols))
        # extension = np.zeros(1*length, dtype=np.int32)
        # for left in range(length):
        #     # chosen_ones = np.random.permutation(length)[:1]
        #     # original_img = pv[img_id, :, :]
        #     # original_label = parsed[img_id]
        #     # # re-label
        #     # new_label =
        #
        #     chosen_ones = np.random.permutation(length)[:1]
        #     extension[left*1:(left+1)*1] = chosen_ones
        #     for j, right in enumerate(chosen_ones):
        #         lim = pv[left,:,:]
        #         rim = pv[right,:,:]
        #         new_im = np.zeros((36,36))
        #         new_im[0:28,0:28] = lim
        #         new_im[6:34,6:34] = rim
        #         new_im[6:28,6:28] = np.maximum(lim[6:28,6:28], rim[0:22,0:22])
        #         multi_data_im =  m.imresize(new_im, (28, 28), interp='nearest')
        #         multi_data[left*1 + j,:,:] = multi_data_im
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
        # , torch.from_numpy(multi_data).view(length,num_rows, num_cols), extension


if __name__ == '__main__':
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import matplotlib.pyplot as plt


    def global_transformer():
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])


    dst = MNIST(root='./Data/MultiMNIST/', train=True, download=True, transform=global_transformer(), multi=True)

    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=4)
    for dat in loader:
        ims = dat[0].view(10, 28, 28).numpy()

        # for t in range(10):
        #     locals()['labs_'+ str(t)] = dat[t+1]
        labs_0 = dat[1][0]
        labs_1 = dat[1][1]
        labs_2 = dat[1][2]
        labs_3 = dat[1][3]
        labs_4 = dat[1][4]
        labs_5 = dat[1][5]
        labs_6 = dat[1][6]
        labs_7 = dat[1][7]
        labs_8 = dat[1][8]
        labs_9 = dat[1][9]
        f, axarr = plt.subplots(2, 5)
        for j in range(5):
            for i in range(2):
                axarr[i][j].imshow(ims[j * 2 + i, :, :], cmap='gray')
                axarr[i][j].set_title('{}_{}'.format(labs_0[j * 2 + i], labs_1[j * 2 + i]))
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()


