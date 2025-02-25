# Sequence 相关类介绍
整个 LLM 的调度和推理都是围绕着 Sequence 进行的。
具体实现见源码 [sequence.py](https://github.com/vllm-project/vllm/blob/main/vllm/sequence.py)

## SequenceData
SequenceData 用于存储与序列相关的数据。这个类主要有三个属性：`prompt_token_ids`（提示词的标记ID）、`output_token_ids`（生成文本的标记ID）和 `cumulative_logprob`（累计对数概率）。
下面是部分伪代码。
```python
class SequenceData:
    """Data associated with a sequence.
    """
    _prompt_token_ids: array # 是一个数组类型，可以理解为 List[int]
    _output_token_ids: array
    _cumulative_logprob: float = 0.0 # 累计对数概率
    
    _num_computed_tokens: int = 0 # 在chunked prefill 调度中，一个 prompt会被拆分，记录已经被计算过的 token 个数

    _num_cached_tokens: int = 0 # 用于 prefill cache 功能，记录哪些token的kv值已经被cache了
    _stage: SequenceStage = SequenceStage.PREFILL # prefill or decode
    _cached_all_token_ids: List[int]

    # It is used to compute mrope_position_ids.
    _mrope_position_delta: Optional[int] = None # 用于位置编码
```

## Sequence
Sequence 主要用于存储序列的数据、状态、输出 token 等，且每个序列有唯一标识，即 seq_id。伪代码如下：
```python
class Sequence:
    """Stores the data, status, and block information of a sequence.
    """

    def __init__(
        self,
        seq_id: int,
        inputs: SingletonInputs, # 为了适配不同类型的输入
        block_size: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.seq_id = seq_id
        self.inputs = SingletonInputsAdapter(inputs)
        self.block_size = block_size

        self.data = SequenceData.from_seqs(self.inputs.prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.status = SequenceStatus.WAITING
        # Input + output tokens
        self.tokens: Optional[List[str]] = None
```
其中的 data 就是上面的 SquenceData ， status 表示当前 Squence 的调度状态，一般是 Wating or Runing，所有句子的初始状态是 Waiting，即等待被调度器调度。

## SquenceGroup 
SquenceGroup 表示一组 Squence，这组 Squence 使用相同的 prompt，这种情形一般发生在 beam search 采样策略中，因此，对于一般的采样，SquenceGroup 中只包含了一个 Squence。
伪代码如下：
```python
class SequenceGroup:
    """A group of sequences that are generated from the same prompt.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        arrival_time: float,
        sampling_params: Optional[SamplingParams] = None,
        lora_request: Optional[LoRARequest] = None,
        pooling_params: Optional[PoolingParams] = None,
        pooled_data: Optional[torch.Tensor] = None,
        encoder_seq: Optional[Sequence] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.first_seq = seqs[0]
        self.arrival_time = arrival_time
        self.is_single_seq = len(seqs) == 1
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}

        self.sampling_params = sampling_params # 采样参数
        self.cached_request_output = None
```
`request_id` 是唯一的标志符，seqs_dict 是一个字典，从 seq_id 映射到 Squence。