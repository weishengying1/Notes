from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import os
import transformers
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel
import json

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class CustomLlama3_1B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=8192,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if schema is not None:
            # Create parser required for JSON confinement using lmformatenforcer
            parser = JsonSchemaParser(schema.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )
            # Output and load valid JSON
            output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
            output = output_dict[0]["generated_text"][len(prompt) :]
            json_result = json.loads(output)

            # Return valid JSON object according to the schema DeepEval supplied
            return schema(**json_result)
        return pipeline(prompt)

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama3 1B"


# 加载模型和分词器
model_name = "/mnt/project/skyllm/weishengying/download/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

custom_llm = CustomLlama3_1B(model=model, tokenizer=tokenizer)

from deepeval.benchmarks import MMLU
benchmark = MMLU(n_shots=0)
benchmark.evaluate(model=custom_llm, batch_size=1)
print("Overall Score:", benchmark.overall_score)
print("Task-specific Scores: ", benchmark.task_scores)