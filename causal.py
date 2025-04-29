from transformer_lens import HookedTransformer
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from data import load_data


def plot_layer_causal(df, fig_dir="causal/layer_causal"):

    yes_no_df = df[df['token'].notna()].copy()
    yes_no_df['token==true_label'] = yes_no_df['token'] == yes_no_df['true_label']
    yes_no_df = yes_no_df.groupby(["layer_id", "token==true_label"]).agg({"delta": "mean"}).reset_index()

    for i in [True, False]:
        sub_df = yes_no_df[yes_no_df[f"token==true_label"] == i]
        sub_df = sub_df.nlargest(5, 'delta')
        print(f"Top K (token == {i}):\n{sub_df.to_string(index=False)}")
    yes_no_heat_data = yes_no_df.pivot(index="token==true_label", columns="layer_id", values="delta")
    yes_no_heat_data = yes_no_heat_data.reindex([True, False])
    plt.figure(figsize=(10, 6))
    
    sns.heatmap(yes_no_heat_data, cmap="coolwarm", cbar=True)
    plt.title("Yes/No Token Causality")
    plt.xlabel("Layer")
    plt.ylabel("Token")
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "yes_no_heatmap.png")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    
    df = df[df["bit"].notna()]
    for year in [1, 2]:
        sub_df = df[df["year"] == year]
    for bit in range(4):
        sub_df = sub_df.groupby(["bit", "layer_id"]).agg({"delta": "mean"}).reset_index()
        sub_df_bit = sub_df[sub_df["bit"] == bit]
        sub_df_bit = sub_df_bit.nlargest(5, 'delta')
        print(f"Top K (Year {year}, Bit {bit}):\n{sub_df_bit.to_string(index=False)}")
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    fig.suptitle("Bit Tokens Causality", fontsize=14)

    for i, year in enumerate([1, 2]):
        ax = axes[i]
        sub_df = df[df["year"] == year]
        sub_df = sub_df.groupby(["layer_id", "bit"]).agg({"delta": "mean"}).reset_index()

        heat_data = sub_df.pivot(index="bit", columns="layer_id", values="delta")

        sns.heatmap(heat_data, ax=ax, cmap="coolwarm", cbar=True, center=0)

        ax.set_title(f"Year {year}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Bit")


    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "bit_heatmap.png")
    plt.savefig(fig_path)

def plot_head_causal(df, fig_dir="causal/head_causal"):

    os.makedirs(fig_dir, exist_ok=True)
    yes_no_df = df[df['token'].notna()].copy()
    yes_no_df['token==true_label'] = yes_no_df['token'] == yes_no_df['true_label']
    

        
    yes_no_df = yes_no_df.groupby(["layer_id", "token==true_label", "head_id"]).agg({"delta": "mean"}).reset_index()
    vmin = yes_no_df['delta'].min()
    vmax = yes_no_df['delta'].max()
    for i in [True, False]:
        sub_df = yes_no_df[yes_no_df[f"token==true_label"] == i]
        sub_df = sub_df.nlargest(5, 'delta')
        print(f"Top K (token == {i}):\n{sub_df.to_string(index=False)}")
    for layer_id in yes_no_df['layer_id'].unique():
        layer_df = yes_no_df[yes_no_df['layer_id'] == layer_id]


        layer_df = layer_df.pivot(index="token==true_label", columns="head_id", values="delta")
        
        layer_df = layer_df.reindex([True, False])
        plt.figure(figsize=(10, 6))
        
        sns.heatmap(layer_df, cmap="coolwarm", cbar=True, vmin=vmin, vmax=vmax)
        
        plt.title(f"Token==True_label Causality - Layer {layer_id}")
        plt.xlabel("Token==True_label")
        plt.ylabel("Head")
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"yes_no_heatmap_layer_{layer_id}.png")
        plt.savefig(fig_path)
        plt.close()

    
    bit_df = df[df["bit"].notna()]
    vmin = bit_df['delta'].min()
    vmax = bit_df['delta'].max()  

    for year in [1, 2]:
        sub_df = bit_df[bit_df["year"] == year]
        for bit in range(4):
            sub_df = sub_df.groupby(["bit", "head_id", "layer_id"]).agg({"delta": "mean"}).reset_index()
            sub_df_bit = sub_df[sub_df["bit"] == bit]
            sub_df_bit = sub_df_bit.nlargest(5, 'delta')
            print(f"Top K (Year {year}, Bit {bit}):\n{sub_df_bit.to_string(index=False)}")

    for layer_id in bit_df['layer_id'].unique():
        layer_df = bit_df[bit_df['layer_id'] == layer_id]
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        fig.suptitle(f"Bit Tokens Causality - Layer {layer_id}", fontsize=14)
        for i, year in enumerate([1, 2]):
            ax = axes[i]
            sub_df = layer_df[layer_df["year"] == year]
            sub_df = sub_df.groupby(["bit", "head_id"]).agg({"delta": "mean"}).reset_index()

                
            sub_df = sub_df.pivot(index="bit", columns="head_id", values="delta")
            
            sns.heatmap(sub_df, ax=ax, cmap="coolwarm", cbar=True, vmin=vmin, vmax=vmax)
            ax.set_title(f"Year {year}")
            ax.set_xlabel("Head")
            ax.set_ylabel("Bit")

        plt.tight_layout() 
        fig_path = os.path.join(fig_dir, f"bit_heatmap_layer_{layer_id}.png")
        plt.savefig(fig_path)
        plt.close()



def layer_causal(model,
                prompts: list,
                true_label: str,
                false_label: str,
                layer_ids=range(32),
                year_pos=(range(22, 26), range(29, 33)),
                yes_no_pos=(37, 41)
                ):
    
    
    def get_metric(logits):
        last_logits = logits[:, -1, :]
        t = model.to_single_token(true_label)
        f = model.to_single_token(false_label)
        return last_logits[:, t] - last_logits[:, f]

    from functools import partial

    def patch(resid, hook, token_pos, method='zero'):
        # print(f"[HOOK] patching token {token_pos} in {hook.name}")
        resid = resid.clone()
        if method == 'zero':
            resid[:, token_pos, :] = 0
        return resid
    
    deltas = []
    with torch.no_grad():
        model.eval()
        model.reset_hooks()
        tokens = model.to_tokens(prompts)
        # print([(i, model.to_string(token)) for i, token in enumerate(tokens[0])])
        baseline_logits = model(tokens)
        baseline_score = get_metric(baseline_logits)
        for layer_id in layer_ids:
            for pi, token_pos in enumerate(yes_no_pos):
                model.reset_hooks()
                patched_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (f"blocks.{layer_id}.hook_resid_post", partial(patch, token_pos=token_pos))
                    ],
                    reset_hooks_end=True,
                    clear_contexts=True
                )
                patched_score = get_metric(patched_logits)
                delta = baseline_score - patched_score
                delta = delta.item()
                deltas.append({
                    "layer_id": layer_id,
                    "delta": delta,
                    "baseline_score": baseline_score.item(),
                    "patched_score": patched_score.item(),
                    'prompt': prompts[0],
                    'true_label': true_label,
                    'false_label': false_label,
                    "token": "Yes" if pi == 0 else "No",
                })
            for year, pos in enumerate(year_pos):
                for bit, token_pos in enumerate(pos):
                    model.reset_hooks()
                    patched_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks = [
                            (f"blocks.{layer_id}.hook_resid_post", partial(patch, token_pos=token_pos))
                        ],
                        reset_hooks_end=True,
                        clear_contexts=True
                    )
                    patched_score = get_metric(patched_logits)
                    delta = baseline_score - patched_score
                    delta = delta.item()
                    deltas.append({
                        "layer_id": layer_id,
                        "bit": bit,
                        "year": year+1,
                        "delta": delta,
                        "baseline_score": baseline_score.item(),
                        "patched_score": patched_score.item(),
                        'prompt': prompts[0],
                        'true_label': true_label,
                        'false_label': false_label,
                    })
    return deltas


def head_causal(model,
                prompts: list,
                true_label: str,
                false_label: str,
                layer_ids: list,
                layer_ids_yn: list,
                head_ids: list,
                year_pos=(range(22, 26), range(29, 33)),
                yes_no_pos=(37, 41)
                ):
    
    
    def get_metric(logits):
        last_logits = logits[:, -1, :]
        t = model.to_single_token(true_label)
        f = model.to_single_token(false_label)
        return last_logits[:, t] - last_logits[:, f]

    from functools import partial

    def patch(result, hook, head_idx, token_pos, method='zero'):
        # print(f"[HOOK] patching head {head_idx} at pos {token_pos} in {hook.name}")
        result = result.clone()
        result[:, token_pos, head_idx, :] = 0
        return result
    
    deltas = []
    with torch.no_grad():
        model.eval()
        model.reset_hooks()
        tokens = model.to_tokens(prompts)
        baseline_logits = model(tokens)
        baseline_score = get_metric(baseline_logits)
        for layer_id in layer_ids_yn:
            for pi, token_pos in enumerate(yes_no_pos):
                for head_id in head_ids:
                    model.reset_hooks()
                    patched_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[
                            (f"blocks.{layer_id}.attn.hook_result", partial(patch, head_idx=head_id, token_pos=token_pos))
                        ],
                        reset_hooks_end=True,
                        clear_contexts=True
                    )
                    patched_score = get_metric(patched_logits)
                    delta = baseline_score - patched_score
                    delta = delta.item()
                    # print(f"Layer {layer_id}, Head {head_id}, Yes/No Token {pi}, Delta: {delta}")
                    deltas.append({
                        "layer_id": layer_id,
                        "head_id": head_id,
                        "delta": delta,
                        "baseline_score": baseline_score.item(),
                        "patched_score": patched_score.item(),
                        'prompt': prompts[0],
                        'true_label': true_label,
                        'false_label': false_label,
                        "token": "Yes" if pi == 0 else "No",
                    })
        for layer_id in layer_ids:
            for year, pos in enumerate(year_pos):
                for bit, token_pos in enumerate(pos):
                    for head_id in head_ids:
                        model.reset_hooks()
                        patched_logits = model.run_with_hooks(
                            tokens,
                            fwd_hooks = [
                                (f"blocks.{layer_id}.attn.hook_result", partial(patch, head_idx=head_id, token_pos=token_pos))
                            ],
                            reset_hooks_end=True,
                            clear_contexts=True
                        )
                        patched_score = get_metric(patched_logits)
                        delta = baseline_score - patched_score
                        delta = delta.item()
                        # print(f"Layer {layer_id}, Head {head_id}, Bit {bit}, Year {year+1}, Delta: {delta}")
                        deltas.append({
                            "layer_id": layer_id,
                            "bit": bit,
                            "year": year+1,
                            "head_id": head_id,
                            "delta": delta,
                            "baseline_score": baseline_score.item(),
                            "patched_score": patched_score.item(),
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
    year_pos = (range(22, 26), range(29, 33))
    answer_pos = (37, 41)
    prompts, labels = load_data("dataset_qa_yesno_150_1800_2020.csv", sample_size=10)
    
    results = pd.DataFrame()
    layers = range(19)
    layers_ids_yn=[4] + list(range(9, 21))
    heads = range(32)
    
    for i, (prompt, label) in tqdm(enumerate(zip(prompts, labels)), total=len(prompts)):
        # deltas = layer_causal(model, [prompt], true_label=label, false_label="Yes" if label == "No" else "No", year_pos=year_pos, layer_ids=layers, yes_no_pos=answer_pos)
        deltas = head_causal(model, [prompt], true_label=label, false_label="Yes" if label == "No" else "No", 
                             year_pos=year_pos, layer_ids=layers, yes_no_pos=answer_pos
                             , layer_ids_yn=layers_ids_yn, head_ids=heads)
        deltas = pd.DataFrame(deltas)
        results = pd.concat([results, deltas], ignore_index=True)
    print(results.head())
    os.makedirs("causal/head_causal", exist_ok=True)
    results.to_csv("causal/head_causal/results.csv", index=False)
    print('==== Head causal ====')
    with open("causal/head_causal/results.csv", "r") as f:
        results = pd.read_csv(f)
    plot_head_causal(results)
    print('==== Layer causal ====')
    with open("causal/layer_causal/results.csv", "r") as f:
        results = pd.read_csv(f)
    plot_layer_causal(results)