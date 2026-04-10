from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prefix = "A long time ago"
input_ids = tokenizer(prefix, return_tensors="pt").input_ids


input_ids = input_ids.to('cuda')
model = model.to('cuda')

outputs = model.generate(
    inputs=input_ids,
    max_new_tokens=15,
    temperature=0.0,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

