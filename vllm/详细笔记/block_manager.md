# 分散管理 kv
分散式管理 kv 已经是几乎所有推理引擎的共识了，这样做的好处就是可以精打细算的节省显存，而没有必要提前根据最大生成长度预留出 kv cache 空间。

## 实际物理内存上的 kv cache
首先看看实际物理内存上的 kv cache 是什么样的

[cache_engine.py](https://github.com/vllm-project/vllm/blob/v0.6.5/vllm/worker/cache_engine.py#L66)

部分代码如下：
```python
def _allocate_kv_cache(
    self,
    num_blocks: int,
    device: str,
) -> List[torch.Tensor]:
    """Allocates KV cache on the specified device."""
    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        num_blocks, self.block_size, self.num_kv_heads, self.head_size)
    pin_memory = is_pin_memory_available() if device == "cpu" else False
    kv_cache: List[torch.Tensor] = []
    for _ in range(self.num_attention_layers):
        kv_cache.append(
            torch.zeros(kv_cache_shape,
                        dtype=self.dtype,
                        pin_memory=pin_memory,
                        device=device))
    return kv_cache
```
内存中的 kv_cache 本质上是由一堆小的 torch tensor 构成的 :
每个 torch tensor 代表一个 layer 中的 kv cache，其 shape 为 : `(2, num_blocks, block_size, num_kv_heads, head_size)`
总的 kv cache shape 为：`(num_attenion_layers, 2, num_blocks, block_size, num_kv_heads, head_size)`

由此可见，最多能够存 `num_blocks * block_size` 个 token 的 kv

## 管理这些 kv cache tensor
提前申请了大量的 zero tensor，这些后面用来存 token 的 kv 值，如何管理这些预申请的 kv tensor，vllm 中在 BlockManager 类中实现的。

先思考下，第一个 token 得 k 值应该存在哪，很显然，第 i 层 layer 的 k 值存在 `kv_cache[i, 0, 0, 0]` 中，v 值存在 `kv_cache[i, 1, 0, 0]` 中，可以看出，前两个位置是由属性确定的。
依次类推，第二个 token 的 kv 存的位置就是： `kv_cache[i, :, 0, 1]`
即 kv_cache 分为 num_blocks 块，每块可以存 block_size 个 token 的 kv 值，按照顺序从 0 到 num_blocks 进行编码。


# NaiveBlock
NaiveBlock 一方面可以理解为是实际某段缓存了 kv 值的物理内存的表示，它描述了这段缓存了 kv 值的内存对应的 token 值、相对物理位置（block_idx）等信息
```python
class NaiveBlock(Block):
    def __init__(self,
                 prev_block: Optional[Block],
                 token_ids: List[int],
                 block_size: int,
                 allocator: BlockAllocator,
                 block_id: Optional[int] = None,
                 _cow_target: Optional[Block] = None,
                 extra_hash: Optional[int] = None):
        self._token_ids: List[int] = []
        self._block_size = block_size
        self._prev_block = prev_block
        self._block_id = block_id
        self._allocator = allocator
        self._cow_target = _cow_target if _cow_target is not None else self

        self._append_token_ids_no_cow(token_ids)
```
如 _token_ids 就是指这段内存中缓存 kv 值对应的 token id，_block_id 就是对这段物理内存位置的描述。即 NaiveBlock 就是一个物理内存的抽象，它描述了物理内存中缓存了 kv 值的 token id，以及物理内存的相对位置。
多个 block 可以 share 同一个 block id，即多个 block 可以共享同一个物理内存。比如不同的 sequence 中，有一些相同的 token id，可以共享同一个物理内存，从而减少显存占用。

block 分为 imutable 和 mutable 两种，其中 immutable 是指 block 是只读的，不能修改，而 mutable 是指 block 是可读写的，可以修改。
如对于一个长度为 33 的 sequence prompt , 假设 block_size = 16, 那么需要申请三个block，第一和第二个 block 是 immutable 的，第三个 block 是 mutable 的， 在后续 decode 过程中生成的新的 token 的 kv 值，会存到第三个 block 中， 等它存满了之后， 也会变成 immutable。

_prev_block 是指当前 block 的前一个 block，如上述长度为 33 的 sequence prompt，第一个 block 的 _prev_block 是 None，后面两个的 prev_block 是前一个 block，这样就把一个 sequence 包含的所有 block 串了起来。方便查找 sequence 的所有 block 信息。

# NaiveBlockAllocator
*NaiveBlockAllocator* 就是管理这段实际物理内存的管理器。*实际的物理内存被表示为一串连续的`block id`*, 它知道哪些内存是已经被申请了，哪些是 free，并负责分配和回收实际内存。这些工作都是基于 block id 完成的。block id 就是实际物理内存的映射。

copy-on-write 操作： 如在 beam search 采样中，初始 prompt 被分成了多个 immutable block 和最后一个 mutable block，为了减少显存占用，它们被多个 sequence 共享,  在后续的 decode 过程中， 不同的分支 decode 的结果可能不同， 因此往最后一个 mutable block append 新的 token 的 kv 值，会导致这个 block 被修改，因此需要做 copy-on-write 操作，即创建申请一个新 block，把原来 block 的内容复制到新 block 中，新 block 有新分配的 block id，新 block 的 _token_ids 是原来 block 的 _token_ids + 新的 token 的kv 值，新 block 的 _prev_block 是原来 block 的 _prev_block。
即对于一些被多个 sequence share 的 mutable block，当需要往里面 append 新的 token 的 kv 值时，会做 copy-on-write 操作。

# BlockPool
BlockPool 里面有很多提前申请好的 Block，这样做的目的是为了减少 Block 这个python 对象的创建和销毁次数，提高代码性能。
一个 block 中如果 _block_id 是空的，仅仅有 _token_ids 是没有意义的，即这个 block 没有实际的物理内存与他对应。所以可以提前申请好很多 block 对象，复用这些 block 对象，仅仅去修改它们的 block_id， token_ids 这些信息等。