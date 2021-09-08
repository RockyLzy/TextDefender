import random
from typing import List


class DataSet:
    def __getitem__(self, item):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented


class SimpleSet(DataSet):
    def __init__(self):
        self.data = [i + 0.1 for i in range(100)]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Sampler:
    def next(self):
        raise NotImplemented

    def finished(self):
        raise NotImplemented

    def reset(self):
        raise NotImplemented


class IndexSampler(Sampler):
    pass


class DataLoader(Sampler):
    pass


class SingleIndexSampler(IndexSampler):
    def __init__(self,
                 max_index,
                 shuffle):
        self.max_index = max_index
        self.shuffle = shuffle
        self.indices = list(range(self.max_index))
        self.__next = 0
        self.reset()

    def next(self) -> int:
        ret = self.indices[self.__next]
        self.__next += 1
        return ret

    def finished(self):
        return self.__next == self.max_index

    def reset(self):
        if self.shuffle:
            random.shuffle(self.indices)


class BatchedIndexSampler(IndexSampler):
    def __init__(self,
                 max_index,
                 shuffle,
                 batch_size,
                 fill_last_batch):
        self.max_index = max_index
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.fill_last_batch = fill_last_batch
        self.indices = list(range(self.max_index))
        self.__next = 0
        self.reset()

    def next(self) -> List[int]:
        if self.__next + self.batch_size > len(self.indices):
            if self.fill_last_batch:
                ret = self.indices[self.max_index - self.batch_size:self.max_index]
            else:
                ret = self.indices[self.__next:self.max_index]
            self.__next = self.max_index
        else:
            ret = self.indices[self.__next:self.__next + self.batch_size]
            self.__next += self.batch_size
        return ret

    def finished(self):
        return self.__next == self.max_index

    def reset(self):
        if self.shuffle:
            random.shuffle(self.indices)


class FixedBatchedDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size, fill_last_batch, shuffle
                 ):
        self.dataset = dataset
        self.sampler = BatchedIndexSampler(max_index=len(self.dataset),
                                           shuffle=shuffle,
                                           batch_size=batch_size,
                                           fill_last_batch=fill_last_batch)

    def next(self):
        return [self.dataset[idx] for idx in self.sampler.next()]

    def size(self):
        return len(self.dataset)

    def finished(self):
        return self.sampler.finished()

    def reset(self):
        self.sampler.reset()


# class DynamicBatchedDataLoader(DataLoader):
#     def __init__(self, dataset, shuffle):
#         self.dataset = dataset
#         self.sampler = SingleIndexSampler(max_index=len(self.dataset),
#                                           shuffle=shuffle)
#
#     def next_batch_index(self):
#         raise NotImplemented
#
#     def next(self):
#         return [self.dataset[idx] for idx in self.next_batch_index()]
#
#     def size(self):
#         return len(self.dataset)
#
#     def finished(self):
#         return self.sampler.finished()
#
#     def reset(self):
#         self.sampler.reset()


if __name__ == '__main__':
    s = SimpleSet()
    dl = FixedBatchedDataLoader(s,
                                batch_size=10,
                                fill_last_batch=10,
                                shuffle=False)
    print(dl.next())
