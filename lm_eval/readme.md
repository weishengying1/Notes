[官网](https://github.com/EleutherAI/lm-evaluation-harness)


可以先在本地启动一个 llm 服务, 使用 vllm or sglang 推理引擎

1. 使用 vllm 推理引擎启动服务（prefill cache 打开会导致服务崩溃）
```bash
python -u -m vllm.entrypoints.openai.api_server  --served_model_name /mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct \
       --model /mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct \
       --gpu-memory-utilization 0.65   --max-model-len 8192  --tensor-parallel-size 1 \
       --pipeline-parallel-size 2 \
       --max-num-seqs 16 --kv-cache-dtype auto --dtype auto \
       --disable-log-request

python -u -m vllm.entrypoints.openai.api_server  --served_model_name /mnt/project/skyllm/weishengying/Notes/transformers/mistal-nemo \
       --model /mnt/project/skyllm/weishengying/Notes/transformers/mistal-nemo \
       --gpu-memory-utilization 0.65   --max-model-len 8192  --tensor-parallel-size 2 \
       --pipeline-parallel-size 1 \
       --max-num-seqs 16 --kv-cache-dtype auto --dtype auto \
       --disable-log-request
```

2. 使用 sglang 推理引擎启动服务
```bash
python -m sglang.launch_server \
       --model-path /mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct \
       --port 8000 \
       --mem-fraction-static  0.65 \
       --disable-custom-all-reduce \
       --load-balance-method round_robin \
       --context-length 8192 \
       --tp-size 2 \
       --enable-mixed-chunk \
       --chunked-prefill-size 512 \
       --kv-cache-dtype auto \
       --schedule-policy lpm \
       --dtype  auto \
       --enable-p2p-check


# 安装 sglang 
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/ #(0.4.0.post2)

# 启动服务 （开关都可以开）
python -m sglang.launch_server \
       --model-path /mnt/project/skyllm/weishengying/Notes/transformers/mistal-nemo \
       --port 8000 \
       --mem-fraction-static  0.65 \
       --disable-custom-all-reduce \
       --load-balance-method round_robin \
       --context-length 8192 \
       --tp-size 2 \
       --enable-mixed-chunk \
       --chunked-prefill-size 512 \
       --kv-cache-dtype auto \
       --schedule-policy lpm \
       --dtype  auto \
       --enable-p2p-check
```

3. 使用 tgi 推理
```bash
text-generation-launcher --model-id /mnt/project/skyllm/weishengying/Notes/transformers/mistal-nemo --sharded true --port 8000 --max-best-of 1 --max-input-tokens 8192 --max-total-tokens 16000 --max-batch-total-tokens 16000 --cuda-graphs "1,2,4,8,16,32"

text-generation-launcher --model-id /mnt/shared/maas/ai_story/llama2_as_def_en_12b_v5_1205 --sharded true --port 8000 --max-best-of 1 --max-input-tokens 8192 --max-total-tokens 9216 --max-batch-total-tokens 8448 --cuda-graphs "1,2,4,8,16,32"
```

然后使用下面脚本（脚本会请求服务）：
```bash
export HF_ENDPOINT=https://hf-mirror.com

lm_eval --model local-completions --tasks mmlu       --model_args model=/mnt/project/skyllm/weishengying/Notes/transformers/mistal-nemo,tokenizer_backend=huggingface,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=5,max_length=8192,temperature=0.6,top_p=0.9  --output_path ./tmp --batch_size=16 --num_fewshot 5

lm_eval --model local-completions --tasks mmlu       --model_args model=/mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct,tokenizer_backend=huggingface,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=5,max_length=8192,temperature=0.6,top_p=0.9  --output_path ./tmp --batch_size=1 --num_fewshot 5

lm_eval --model local-completions --tasks mmlu       --model_args model=/mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct,tokenizer_backend=huggingface,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=5,max_length=4096,temperature=0.6,top_p=0.9  --output_path ./tmp --batch_size=1 --num_fewshot 0

lm_eval --model local-completions --tasks ifeval     --model_args model=/mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct,tokenizer_backend=huggingface,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=5,max_length=8192,temperature=0.6,top_p=0.9  --output_path ./tmp --batch_size=1 --num_fewshot 0

lm_eval --model local-completions --tasks gsm8k     --model_args model=/mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct,tokenizer_backend=huggingface,base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=5,max_length=8192,temperature=0.6,top_p=0.9  --output_path ./tmp --batch_size=1 --num_fewshot 8
```

# 使用 transformer 推理写的服务
```bash
lm_eval --model local-completions --tasks mmlu       --model_args model=/mnt/project/skyllm/weishengying/Notes/transformers/mistal-nemo,tokenizer_backend=huggingface,base_url=http://localhost:8000/generate,num_concurrent=1,max_retries=5,max_length=8192,top_k=5,top_p=0.85  --output_path ./tmp --batch_size=1 --num_fewshot 0
```


也可以直接使用线上的服务：
```bash
export OPENAI_API_KEY=YTBiZjBiNzg5ZWMxNDE4NmI1MWJiYzkzMDRlZTJmYjAxNDBhOTNiMw==
export BASE_URL=http://1893706806886638.cn-beijing.pai-eas.aliyuncs.com/api/predict/prod_llama2_as_def_en_12b_v5_pressure_t_1204/v1/completions
export HF_ENDPOINT=https://hf-mirror.com
lm_eval --model local-completions --tasks gsm8k --model_args model=llama2_as_def_en_12b_v5_pressure_t_1204,tokenizer_backend=None,tokenized_requests=False,base_url=${BASE_URL},num_concurrent=1,max_retries=1 --gen_kwargs temperature=1.0,top_k=5 --batch_size=4 --num_fewshot 1
```
