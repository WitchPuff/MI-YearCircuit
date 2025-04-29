from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer


# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# model  = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# print(tokenizer)

model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = model.to('cuda')
print(model)
n_heads = model.cfg.n_heads
model.cfg.use_attn_result = True
model.set_use_attn_result(True)
print(f"Number of attention heads per layer: {n_heads}")
