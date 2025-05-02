import torch
import torch.nn.functional as F
import pandas as pd
from functools import partial
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data import load_data
from tqdm import tqdm
from collections import Counter

def find_answer_circuit(
        model,
        prompt: str,
        layer_ids: list,
        true_label: str,
        false_label: str,
        answer_pos: int | None = None,
        head_limit: int | None = None,
        device='cuda'
):

    def get_metric(logits):
        last_logits = logits[:, -1, :]
        t = model.to_single_token(true_label)
        f = model.to_single_token(false_label)
        return last_logits[:, t] - last_logits[:, f]
    
    
    tokens = model.to_tokens(prompt).to(device)
    if answer_pos is None:
        answer_pos = tokens.shape[-1] - 1

    with torch.no_grad():
        base_logits, base_cache = model.run_with_cache(tokens)


    results = []

    for layer_id in layer_ids:
        n_heads = model.cfg.n_heads if head_limit is None else head_limit
        for head_id in range(n_heads):
            
            def patch(result, hook, head_idx=head_id, tok_pos=answer_pos):
                result = result.clone()
                result[:, tok_pos, head_idx, :] = 0
                return result

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer_id}.attn.hook_result",
                                partial(patch, head_idx=head_id))],
                    reset_hooks_end=True, clear_contexts=True
                )
            patched_score = get_metric(patched_logits)
            base_score = get_metric(base_logits)
            delta = base_score - patched_score
            if delta <= 0:
                continue

            attn = base_cache[f"blocks.{layer_id}.attn.hook_pattern"][0, head_id, answer_pos]
            
            top_idx = attn.topk(5).indices.tolist()
            top_tokens = [model.to_string(int(tokens[0, i])) for i in top_idx]

            head_out = base_cache[f"blocks.{layer_id}.attn.hook_result"][0, answer_pos, head_id]
            true_label_dir  = model.W_U[:, model.to_single_token(true_label)]
            false_label_dir   = model.W_U[:,  model.to_single_token(false_label)]
            cos_true  = F.cosine_similarity(head_out, true_label_dir, dim=0).item()
            cos_false   = F.cosine_similarity(head_out, false_label_dir , dim=0).item()

            results.append(dict(
                layer_id=layer_id, head_id=head_id, delta=delta.item(),
                attn_top=" | ".join(top_tokens),
                cos_true=round(cos_true, 3),
                cos_false =round(cos_false , 3),
                prompt=prompt,
                true_label=true_label,
                false_label=false_label,
            ))


    return results

def plot_answer_head(df, metrics=["delta", "cos_true"]):

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.suptitle("Answer Circuit", fontsize=14)
    # vmax = df[metrics].max().max()
    # vmin = df[metrics].min().min()
    for i, ax in enumerate(axes):
        metric = metrics[i]
        mdf = df.pivot(index='head_id', columns='layer_id', values=metric)
        sns.heatmap(mdf, ax=ax, cmap="coolwarm", cbar=True)

        ax.set_title(f"Year {metric}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
    plt.tight_layout()
    os.makedirs("circuit", exist_ok=True)
    plt.savefig(f"circuit/answer_head.png")


def count_attn_tokens(df):
    rows = []
    for _, row in df.iterrows():
        layer = row['layer_id']
        head = row['head_id']
        attn_top = row['attn_top']
        tokens = [tok.strip() for tok in attn_top.split("|")]
        for token in tokens:
            rows.append((layer, head, token))
    
    counter = Counter(rows)
    
    data = []
    for (layer, head, token), count in counter.items():
        data.append({
            "layer_id": layer,
            "head_id": head,
            "token": token,
            "count": count
        })
    
    return pd.DataFrame(data)

def plot_attn_tokens(df):

    pivot = df.pivot_table(index="token", columns=["layer_id", "head_id"], values="count", fill_value=0)

    pivot.columns = [f"L{l}-H{h}" for l, h in pivot.columns]

    full_columns = [f"L{l}-H{h}" for l in range(11, 32) for h in range(32)]
    pivot = pivot.reindex(columns=full_columns, fill_value=0)

    plt.figure(figsize=(20, 8))
    plt.xticks(rotation=45)
    sns.heatmap(pivot, cmap="Reds", cbar=True, annot=False)
    plt.title("Top5 Most Attended Token Frequency by Answer Token Position")
    plt.xlabel("Layer - Head")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.savefig("circuit/attn_tokens.png")

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model.cfg.use_attn_result = True
    model.set_use_attn_result(True)
    model = model.to('cuda')
    prompts, labels = load_data("dataset_qa_yesno_150_1800_2020.csv", sample_size=10)
    
    results = pd.DataFrame()
    layers_ids_yn=range(11, 32)
    results = []
    ret_df = pd.DataFrame()
    for i, (prompt, label) in tqdm(enumerate(zip(prompts, labels)), total=len(prompts)):
        results += find_answer_circuit(
            model=model,
            prompt=prompt,
            layer_ids=layers_ids_yn,
            true_label=label,
            false_label="Yes" if label == "No" else "No",
        )
    ret_df = pd.DataFrame(results)
    os.makedirs("circuit", exist_ok=True)
    ret_df.to_csv("circuit/answer_head.csv", index=False)
    with open("circuit/answer_head.csv", "r") as f:
        df = pd.read_csv(f)
    df = df.groupby(['layer_id', 'head_id']).agg({"cos_true": "mean", "delta": "mean"}).reset_index()
    plot_answer_head(df, metrics=["delta", "cos_true"])
    print('==== Answer Circuit ====')
    print("\n===== Top heads writing answer residual (cos_true desc) =====")
    cos_df = df.sort_values("cos_true", ascending=False).head(5).reset_index(drop=True)
    print(cos_df.head())
    print("\n===== Top heads writing answer residual (delta desc)=====")
    delta_df = df.sort_values("delta", ascending=False).head(5).reset_index(drop=True)
    print(delta_df.head())