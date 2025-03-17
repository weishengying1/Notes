# 1. _passed_delay 控制
https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py#L1078

> 设计的目的就是为了减少 prefill 调度的频率，增加 decode 的频率，在 TTFT 和 TPOT 之间平衡。
> 实现上也很简单： 两次调度间隔时间 = 当前时间 - 上一次调度的时间
> 两次调度间隔时间 < （当前时间 - 等待队列中请求的最早到达时间） * factor
> 如果上述条件，就跳过这次调度，但是实现却是：
> 两次调度间隔时间 * factor < （当前时间 - 等待队列中请求的最早到达时间）
> 个人觉得不对。
> https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py#L1842

# _get_num_new_uncached_and_cached_tokens
for chunked prefill
num_computed = chunked_size * n(1,2,...)
|------------------------- seq_len ---------------------------|
|-------- num_computed ----------|---- all_num_new_tokens ----|

# 3. _get_num_new_uncached_and_cached_tokens
https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py#L1945

num_cached_tokens_seq 一定是 block_size 的倍数，所以只有 chunked prefill 才会有这个情况。


# 2. _schedule_chunked_prefill
https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py#L1325

chunked prefill 调度已经成为主流。