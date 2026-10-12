import math
import torch
from torch.utils.data.distributed import DistributedSampler
import random

class InterleaveSampler(DistributedSampler):
    def __init__(
            self,
            dataset,
            batch_size,
            num_replicas,
            rank,
            seed=0,
            shuffle=True
        ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        self.batch_size = batch_size
        self.rho = dataset.rho
        self.num_voc = dataset.num_voc
        self.num_gen = len(dataset) - self.num_voc
        self.gen_per_batch = int(self.batch_size * self.rho)
        self.voc_per_batch = self.batch_size - self.gen_per_batch
        if self.rho != 1.0:
            self.num_batches = math.ceil(self.num_voc / self.voc_per_batch)  # ensure all voc data is seen once per epoch
        else:
            self.num_batches = math.ceil(self.num_gen / self.gen_per_batch)  # only replay data

        # update self.num_batches to be divisible by num_replicas
        if self.num_batches % self.num_replicas != 0:
            self.num_batches = math.ceil(self.num_batches / self.num_replicas) * self.num_replicas

        self.num_samples = math.ceil((self.batch_size * self.num_batches) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas  # == self.batch_size * self.num_batches
        if self.rho != 1.0:
            self.repeat_replay = math.ceil((self.total_size - self.num_voc) / self.num_gen)
        else:
            self.repeat_replay = math.ceil(self.total_size / self.num_gen)

    def __iter__(self):
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            voc_inds = torch.randperm(self.num_voc, generator=g).tolist()  # type: ignore[arg-type]
            gen_inds = []
            for i in range(self.repeat_replay):
                g.manual_seed(self.seed + self.epoch + 100000*i) # avoid repeated seeds
                gen_inds += (torch.randperm(self.num_gen, generator=g) + self.num_voc).tolist()  # type: ignore[arg-type]

        else:
            voc_inds = list(range(self.num_voc))  # type: ignore[arg-type]
            gen_inds = []
            for i in range(self.repeat_replay):
                gen_inds += list(range(self.num_voc, self.num_voc + self.num_gen, 1))  # type: ignore[arg-type]
        if self.rho != 1.0:
            gen_inds = gen_inds[:(self.total_size - self.num_voc)]  # trim to the total size
        else:
            gen_inds = gen_inds[:self.total_size]  # trim to the total size
            voc_inds = []
        if len(voc_inds + gen_inds) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(voc_inds + gen_inds)}) does not match total_size ({self.total_size})"
            )
        full_inds = []
        for i in range(self.num_batches):
            if i < self.num_batches - 1:
                batch = voc_inds[i*self.voc_per_batch : (i+1)*self.voc_per_batch]
                batch += gen_inds[i*self.gen_per_batch : (i+1)*self.gen_per_batch]
            else:
                batch = voc_inds[i*self.voc_per_batch : (i+1)*self.voc_per_batch]
                b_len = len(batch)
                batch += gen_inds[i*self.gen_per_batch : i*self.gen_per_batch + (self.batch_size - b_len)]  # fill last batch with replay if not enough real data
            if self.shuffle:
                random.shuffle(batch)
            full_inds.append(batch)

        # subsample
        indices = []
        for i in range(self.rank, len(full_inds), self.num_replicas):
            indices += full_inds[i]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        # pyrefly: ignore [bad-return]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch