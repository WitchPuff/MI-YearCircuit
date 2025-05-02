import torch
import torch.nn.functional as F
import pandas as pd
from transformer_lens import HookedTransformer
import os
from data import load_data
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_attn_score(cache, pos1, pos2, layer_id, head_id):
    pattern = cache[f"blocks.{layer_id}.attn.hook_pattern"][0, head_id]  # [Q, K]
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    # score_a2b = pattern[pos1, pos2].item() # the previous a can't look at the latter b for causal modelling.
    score_b2a = pattern[pos2, pos1].item()
    return score_b2a

def get_diff_similarity(cache, pos1, pos2, layer_id, head_id):
    out_a = cache[f"blocks.{layer_id}.attn.hook_result"][0, pos1, head_id]
    out_b = cache[f"blocks.{layer_id}.attn.hook_result"][0, pos2, head_id]
    diff = cache[f"blocks.{layer_id}.hook_resid_post"][0, pos1] - cache["blocks.0.hook_resid_post"][0, pos2]
    sim_a = F.cosine_similarity(out_a, diff, dim=0).item()
    sim_b = F.cosine_similarity(out_b, diff, dim=0).item()
    return sim_a, sim_b

def get_qk_diff(cache, pos1, pos2, layer_id, head_id):
    q = cache[f"blocks.{layer_id}.attn.hook_q"][0, :, head_id]
    k = cache[f"blocks.{layer_id}.attn.hook_k"][0, :, head_id]

    qk = q @ k.T

    logits_a = qk[pos1]
    logits_b = qk[pos2]

    sim = F.cosine_similarity(logits_a, -logits_b, dim=0).item()

    return sim



    



def find_compare_circuit(model,
                prompts: list,
                true_label: str,
                false_label: str,
                layer_ids=range(32),
                head_ids=range(32),
                year_pos=(range(22, 26), range(29, 33)),
                ):


    deltas = []
    with torch.no_grad():
        model.eval()
        model.reset_hooks()
        # tokens = model.to_tokens(prompts)
        # _, cache = model.run_with_cache(tokens)
        for layer_id in layer_ids:
            for head_id in head_ids:
                for bit in range(4):
                    pos1 = year_pos[0][bit]
                    pos2 = year_pos[1][bit]
                    delta, cache = compare_causal(model,
                            prompts=prompts,
                            layer_id=layer_id,
                            head_id=head_id,
                            bit=bit,
                            true_label=true_label,
                            false_label=false_label,
                            year_pos=year_pos
                            )
                    score_b2a = get_attn_score(cache, pos1, pos2, layer_id, head_id)
                    sim_a, sim_b = get_diff_similarity(cache, pos1, pos2, layer_id, head_id)
                    qk_diff = get_qk_diff(cache, pos1, pos2, layer_id, head_id)
                    # print(f"Layer {layer_id}, Head {head_id}, Bit {bit}, Year {year+1}, Delta: {score_b2a}")
                    deltas.append({
                        "layer_id": layer_id,
                        "bit": bit,
                        "head_id": head_id,
                        "delta": delta,
                        "score_b2a": score_b2a,
                        "sim_a": sim_a,
                        "sim_b": sim_b,
                        "qk_diff": qk_diff,
                        'prompt': prompts[0],
                        'true_label': true_label,
                        'false_label': false_label,
                    })


    return deltas

def compare_causal(model,
                prompts: list,
                bit: int,
                true_label: str,
                false_label: str,
                layer_id,
                head_id,
                year_pos=(range(22, 26), range(29, 33))
                ):
    def get_metric(logits):
        last_logits = logits[:, -1, :]
        t = model.to_single_token(true_label)
        f = model.to_single_token(false_label)
        return last_logits[:, t] - last_logits[:, f]

    from functools import partial

    def patch_head_result_swap(value, hook, pos1, pos2, head_id, vec_a, vec_b):
        value[0, pos1, head_id] = vec_b
        value[0, pos2, head_id] = vec_a
        return value
    with torch.no_grad():
        model.eval()
        model.reset_hooks()
        tokens = model.to_tokens(prompts)
        # print([(i, model.to_string(token)) for i, token in enumerate(tokens[0])])
        baseline_logits, cache = model.run_with_cache(tokens, return_type="logits")
        baseline_score = get_metric(baseline_logits)
        pos1 = year_pos[0][bit]
        pos2 = year_pos[1][bit]
        vec_a = cache[f"blocks.{layer_id}.attn.hook_result"][0, pos1, head_id].detach().clone()
        vec_b = cache[f"blocks.{layer_id}.attn.hook_result"][0, pos2, head_id].detach().clone()
        hook_fn = partial(
            patch_head_result_swap,
            pos1=pos1, pos2=pos2,
            head_id=head_id,
            vec_a=vec_a, vec_b=vec_b
        )
        patched_logits = model.run_with_hooks(tokens, return_type="logits",
                            fwd_hooks=[(f"blocks.{layer_id}.attn.hook_result", hook_fn)],
                            reset_hooks_end=True,
                            clear_contexts=True
                            )
        patched_score = get_metric(patched_logits)
        delta = baseline_score - patched_score
        delta = delta.item()
    return delta, cache



def plot_compare_head(df, caption='', metric='score_b2a', vmax=None, vmin=None):


    # metrics = ["score_b2a", "sim_a", "sim_b", "qk_diff", "diff(sim_a, sim_b)"]
    if metric == "diff(sim_a, sim_b)":
        df[metric] = 1 - abs(df["sim_a"] + df["sim_b"]) / (abs(df["sim_a"]) + abs(df["sim_b"]) + 1e-6)

    mdf = df.pivot(index='head_id', columns='layer_id', values=metric)
    plt.figure(figsize=(14, 8))
    sns.heatmap(mdf, cmap="coolwarm", cbar=True, vmax=vmax, vmin=vmin)
    plt.title(f"{caption}_{metric}")
    plt.xlabel("Layer")
    plt.ylabel("Head")
    plt.tight_layout()
    plt.savefig(f"circuit/compare_head_{caption}_{metric}.png")

if __name__ == "__main__":
    # model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    # model.cfg.use_attn_result = True
    # model.set_use_attn_result(True)
    # model = model.to('cuda')
    # csv_path = "dataset_qa_yesno_150_1800_2020_True.csv"
    # sample_size=10
    # prompts, labels = load_data(csv_path, sample_size=sample_size)
    
    # results = pd.DataFrame()
    # results = []
    # ret_df = pd.DataFrame()
    # for i, (prompt, label) in tqdm(enumerate(zip(prompts, labels)), total=len(prompts)):
    #     results += find_compare_circuit(
    #         model=model,
    #         prompts=[prompt],
    #         true_label=label,
    #         false_label="Yes" if label == "No" else "No",
    #     )
    # ret_df = pd.DataFrame(results)
    # ret_dir = os.path.join("circuit", '_'.join(csv_path.split("/")[-3:]).split(".")[0]+f"_{sample_size}")
    # os.makedirs(ret_dir, exist_ok=True)
    # ret_df.to_csv(os.path.join(ret_dir, "compare_circuit.csv"), index=False)

    with open(os.path.join('circuit', "compare_circuit.csv"), "r") as f:
        df = pd.read_csv(f)
    metrics = ['delta', "score_b2a", "sim_a", "sim_b", "qk_diff"]
    df = df.groupby(['layer_id', 'head_id', 'bit']).agg({metric: "mean" for metric in metrics}).reset_index()
    metrics.append("diff(sim_a, sim_b)")
    for bit in range(4):
        bit_df = df[df['bit'] == bit]
        for metric in metrics:
            # if metric is not "diff(sim_a, sim_b)":
            #     vmax = df[metric].max()
            #     vmin = df[metric].min()
            print(f"\n===== Top heads compare heads (Bit{bit}, {metric} desc) =====")
            plot_compare_head(bit_df, caption=f"Digit{bit}", metric=metric)
            ret_df = bit_df.sort_values(metric, ascending=False).head(10).reset_index(drop=True)
            print(ret_df.head())
