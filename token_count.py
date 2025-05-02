from transformers import AutoTokenizer
from collections import Counter


tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
print(tokenizer)


years = list(range(1000, 2020))
for year in years:
    year = str(year)
    tokens = tokenizer.tokenize(year)
    ids = tokenizer.encode(year, add_special_tokens=False)

    print(f"Tokens for '{year}': {tokens}")
    print(f"Token IDs: {ids}, length: {len(ids)}") # e.g. Tokens for '1789': ['▁', '1', '7', '8', '9']
token_lens = Counter(len(tokenizer.encode(str(y))) for y in years)
print(token_lens) # Counter({5: 1020})
