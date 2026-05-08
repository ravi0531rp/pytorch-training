# Streaming Datasets

## When to Use
- Dataset > available RAM
- Data stored on object storage (S3, GCS, HDFS)
- Need to start training before full dataset is downloaded
- Online data augmentation pipelines

## WebDataset (Recommended for Large Scale)

```python
import webdataset as wds

def build_streaming_loader(urls, batch_size, num_workers=8):
    """
    urls: list of shards e.g. ["s3://bucket/shard-{000000..001000}.tar"]
    Each shard is a tar file containing sample pairs: sample.jpg, sample.json
    """
    dataset = (
        wds.WebDataset(urls, resampled=True, shardshuffle=True)
        .shuffle(1000)                      # Buffer shuffle
        .decode("pil")                      # Decode images
        .to_tuple("jpg", "json")            # Unpack keys
        .map_tuple(transform, label_fn)     # Apply transforms
        .batched(batch_size, partial=False)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,      # Batching done in dataset
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader

# For distributed: each worker gets disjoint shard subsets
# WebDataset handles this automatically with resampled=True
```

## IterableDataset (Custom Streaming)

```python
from torch.utils.data import IterableDataset
import math

class StreamingDataset(IterableDataset):
    def __init__(self, data_source, transform=None):
        self.data_source = data_source  # e.g., list of file paths or URLs
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process: iterate all
            yield from self._iterate(self.data_source)
        else:
            # Multi-process: split shards across workers
            per_worker = math.ceil(len(self.data_source) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.data_source))
            yield from self._iterate(self.data_source[start:end])

    def _iterate(self, sources):
        for source in sources:
            for sample in self._load(source):
                if self.transform:
                    sample = self.transform(sample)
                yield sample

    def _load(self, source):
        # Implement your loading logic here
        raise NotImplementedError
```

## Key Rules for Streaming

1. **Never use `len()` on IterableDataset** — it's undefined. Estimate steps from dataset size / batch size.
2. **Shard your data** — each worker should get different shards, not the same data.
3. **Set `drop_last=True`** — streaming datasets rarely produce perfect multiples of batch size.
4. **No DistributedSampler** — handle rank-based sharding inside the dataset.
5. **Buffer shuffle** — use in-memory shuffle buffer of 1000-10000 samples to approximate random access.
