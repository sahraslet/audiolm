from transformers import AutoTokenizer
from audiolm.functional import compute_wer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
predicted = "The cat is walking on the roof"
reference = "The cat is walking on the roof"

pred_input = tokenizer(predicted, return_tensors="pt").input_ids
ref_input = tokenizer(reference, return_tensors="pt").input_ids

print("Calculating WER score...")
print("Result:", compute_wer(pred_input, ref_input, tokenizer))