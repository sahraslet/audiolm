from transformers import AutoTokenizer
from audiolm.functional import compute_bleu

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
predicted = "The cat is walking on the roof"
reference = "The cat is walking on the carpet"

pred_input = tokenizer(predicted, return_tensors="pt").input_ids
ref_input = tokenizer(reference, return_tensors="pt").input_ids

print("Calculating BLEU score...")
print("Result:", compute_bleu(pred_input, ref_input, tokenizer))

