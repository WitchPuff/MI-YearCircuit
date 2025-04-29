from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformer_lens import HookedTransformer
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
from sklearn.linear_model import Ridge

def ridge(x, y: list, caption='', alpha=1e-4):
    x = x.cpu().numpy()
    reg = Ridge(alpha=alpha).fit(x, y)
    score = reg.score(x, y)
    v_time = reg.coef_ / np.linalg.norm(reg.coef_)
    print(f"[{caption}] R² =", score)
    print(f"[{caption}] Coefficients =", v_time)
    return score, v_time

def visualize_r2(results, caption='', result_dir='probing'):

    os.makedirs(result_dir, exist_ok=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    layers = list(results.keys())
    token_positions = list(results[layers[0]].keys())
    token_positions.remove('avg')

    for pos in token_positions:
        scores = [results[layer][pos] for layer in layers]
        plt.plot(layers, scores, marker='o', label=f'Token Position {pos}')

    avg_scores = [results[layer]['avg'] for layer in layers]
    plt.plot(layers, avg_scores, marker='x', linestyle='--', color='black', label='Average')

    plt.title(f"R² Scores Across Layers for Token Positions and Average({caption})")
    plt.xlabel("Layer")
    plt.ylabel("R² Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"ridge_{caption}.png"))

def visualize_vtime(x_avg, v_time_avg, x_pos, vtimes, y, caption='', result_dir='probing'):
    os.makedirs(result_dir, exist_ok=True)

    x_avg = x_avg.cpu().numpy()
    x_pos = x_pos.cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    x_avg = x_avg @ v_time_avg
    ax1.scatter(x_avg, y, alpha=0.8, marker='x', label='Average', color='black')

    lr = LinearRegression().fit(x_avg.reshape(-1, 1), y)
    proj_years = lr.predict(x_avg.reshape(-1, 1))
    ax2.scatter(proj_years, y, color='black', alpha=0.8, label='Average', marker='x')

    for i in range(x_pos.shape[1]):
        x_pos_i = x_pos[:, i, :] @ vtimes[i]
        ax1.scatter(x_pos_i, y, alpha=0.5, label=f'Token {i}', marker='o')
        lr = LinearRegression().fit(x_pos_i.reshape(-1, 1), y)
        proj_years = lr.predict(x_pos_i.reshape(-1, 1))
        ax2.scatter(proj_years, y, alpha=0.5, label=f'Token {i}', marker='o')

    ax1.set_title(f"Time Direction Projection ({caption})")
    ax1.set_xlabel("Time Direction")
    ax1.set_ylabel("Years")
    ax1.legend()

    ax2.set_title(f"Time Direction Projection Years ({caption})")
    ax2.set_xlabel("Projected Years")
    ax2.set_ylabel("Years")
    ax2.legend()

    plt.savefig(os.path.join(result_dir, f"vtime_{caption}.png"))


if __name__ == "__main__":

    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    # model  = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    # model = AutoModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct", output_hidden_states=True)
    model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    print(model)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    min_year = 1000
    max_year = 2020
    dataset = pd.read_csv(f"dataset_qa_yesno_150_{min_year}_{max_year}.csv")
    years1 = dataset['year1'].tolist()
    years2 = dataset['year2'].tolist()
    years = (years1, years2)

    year_pos = (range(22, 26), range(29, 33))
    
    

    with torch.no_grad():

        tokens = model.to_tokens(dataset['prompt'].tolist())
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith("hook_resid_post"))

        results = {'Year1': {}, 'Year2': {}}

        for layer in range(model.cfg.n_layers):
            results['Year1'][layer] = {}
            results['Year2'][layer] = {}

            features = cache[f"blocks.{layer}.hook_resid_post"]

            for year_id, pos in enumerate(year_pos):
                vtimes = []
                year_key = f"Year{year_id+1}"

                print(pos)
                token_ids = tokens[0, pos[0]: pos[-1]+1]
                print(model.to_str_tokens(token_ids))

                res_pos = features[:, pos[0]: pos[-1]+1, :]
                res_avg = features.mean(dim=1)

                score, v_time_avg = ridge(res_avg, years[year_id], caption=f'Residual layer {layer} Avg {year_key}')
                results[year_key][layer]['avg'] = score

                for i in range(res_pos.shape[1]):
                    res_pos_i = res_pos[:, i, :]
                    score, v_time = ridge(res_pos_i, years[year_id], caption=f'Residual layer {layer} Token {i} {year_key}')
                    vtimes.append(v_time)
                    results[year_key][layer][i] = score

                visualize_vtime(res_avg, v_time_avg, res_pos, vtimes, years[year_id], 
                                caption=f"Residual layer {layer} {year_key}",
                                result_dir=f"probing/{min_year}_{max_year}")
        print(results)
        with open(os.path.join(f"probing/{min_year}_{max_year}", "results_probing.json"), "w") as f:
            json.dump(results, f, indent=4)
    with open(os.path.join(f"probing/{min_year}_{max_year}", "results_probing.json"), "r") as f:
        results = json.load(f)
        visualize_r2(results['Year1'], caption='Year1', result_dir=f"probing/{min_year}_{max_year}")
        visualize_r2(results['Year2'], caption='Year2', result_dir=f"probing/{min_year}_{max_year}")
        




