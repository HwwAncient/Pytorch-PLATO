"""
DataLoader class
"""

import math

from plato.args import str2bool
from plato.data.batch import batch
from plato.data.sampler import RandomSampler
from plato.data.sampler import SequentialSampler
from plato.data.sampler import SortedSampler


class DataLoader(object):
    """ Implement of DataLoader. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--shuffle", type=str2bool, default=True)
        group.add_argument("--sort_pool_size", type=int, default=0)
        return group

    def __init__(self, dataset, hparams, collate_fn=None, sampler=None, is_test=False, is_train=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sort_pool_size = hparams.sort_pool_size

        if sampler is None:
            if hparams.shuffle and not is_test:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if self.sort_pool_size > 0 and not is_test:
            sampler = SortedSampler(sampler, self.sort_pool_size)

        def reader():
            for idx in sampler:
                yield idx

        self.reader = batch(reader, batch_size=hparams.batch_size, drop_last=False)
        self.num_batches = math.ceil(len(dataset) / hparams.batch_size)

        return

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """
        __iter__函数在for循环遍历本类实例之前调用一次，返回一个新的生成器
        此处生成器以类的形式而非函数的形式产生，是为了暴露生成器中额外的状态
        1. 通过Sampler产生batch data index：[1, 2, 3]
        2. 通过Dataset产生batch data：[[x1, y1], [x2, y2], [x3, y3]]
        3. 通过collate_fn重新组织batch data: [[x1, x2, x3], [y1, y2, y3]]
        """
        for batch_indices in self.reader():
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)
