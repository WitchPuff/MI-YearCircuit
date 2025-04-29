import torch
import torch.nn.functional as F
import pandas as pd
from transformer_lens import HookedTransformer
import os
from data import load_data
from tqdm import tqdm


def get_attn_score(cache, pos_a, pos_b, layer_id, head_id):
    pattern = cache[f"blocks.{layer_id}.attn.hook_pattern"][0, head_id]  # [Q, K]
    if pos_a > pos_b:
        pos_a, pos_b = pos_b, pos_a
    # score_a2b = pattern[pos_a, pos_b].item() # the latter b can't see previous a
    score_b2a = pattern[pos_b, pos_a].item()
    return score_b2a

def get_diff_similarity(cache, pos_a, pos_b, layer_id, head_id):
    out_a = cache[f"blocks.{layer_id}.attn.hook_result"][0, pos_a, head_id]
    out_b = cache[f"blocks.{layer_id}.attn.hook_result"][0, pos_b, head_id]
    diff = cache[f"blocks.{layer_id}.hook_resid_post"][0, pos_a] - cache["blocks.0.hook_resid_post"][0, pos_b]
    sim_a = F.cosine_similarity(out_a, diff, dim=0).item()
    sim_b = F.cosine_similarity(out_b, diff, dim=0).item()
    return sim_a, sim_b
    
def find_compare_circuit(model,
                prompts: list,
                true_label: str,
                false_label: str,
                layer_ids: list,
                head_ids=range(32),
                year_pos=(range(22, 26), range(29, 33)),
                ):


    deltas = []
    with torch.no_grad():
        model.eval()
        model.reset_hooks()
        tokens = model.to_tokens(prompts)
        _, cache = model.run_with_cache(tokens)
        for layer_id in layer_ids:
            for bit in range(4):
                for head_id in head_ids:
                    score_b2a = get_attn_score(cache, year_pos[0][bit], year_pos[1][bit], layer_id, head_id)
                    sim_a, sim_b = get_diff_similarity(cache, year_pos[0][bit], year_pos[1][bit], layer_id, head_id)
                    # print(f"Layer {layer_id}, Head {head_id}, Bit {bit}, Year {year+1}, Delta: {score_b2a}")
                    deltas.append({
                        "layer_id": layer_id,
                        "bit": bit,
                        "head_id": head_id,
                        "score_b2a": score_b2a,
                        "sim_a": sim_a,
                        "sim_b": sim_b,
                        'prompt': prompts[0],
                        'true_label': true_label,
                        'false_label': false_label,
                    })
    return deltas


if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model.cfg.use_attn_result = True
    model.set_use_attn_result(True)
    model = model.to('cuda')
    prompts, labels = load_data("dataset_qa_yesno_150_1000_2020.csv", sample_size=None)
    
    results = pd.DataFrame()
    layers_ids=range(32)
    results = []
    ret_df = pd.DataFrame()
    for i, (prompt, label) in tqdm(enumerate(zip(prompts, labels)), total=len(prompts)):
        results += find_compare_circuit(
            model=model,
            prompts=[prompt],
            layer_ids=layers_ids,
            true_label=label,
            false_label="Yes" if label == "No" else "No",
        )
    ret_df = pd.DataFrame(results)
    os.makedirs("circuit", exist_ok=True)
    ret_df.to_csv("circuit/compare_circuit.csv", index=False)
    with open("circuit/compare_circuit.csv", "r") as f:
        df = pd.read_csv(f)
    df = df.groupby(['layer_id', 'head_id', 'bit']).agg({"score_b2a": "mean",
                                                  "sim_a": "mean",
                                                  "sim_b": "mean"}).reset_index()
    for bit in range(4):
        bit_df = df[df['bit'] == bit]
        for metric in ['score_b2a', 'sim_a', 'sim_b']:
            print(f"\n===== Top heads compare heads (Bit{bit}, {metric} desc) =====")
            ret_df = bit_df.sort_values(metric, ascending=False).head(5).reset_index(drop=True)
            print(ret_df.head())