from tensorrt_llm import LLM, SamplingParams

prompts = ["""[INST]who are you?/INST]"""]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/mnt/shared/maas/ai_story/llama3_as_en_8b_v5v1_1108")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")