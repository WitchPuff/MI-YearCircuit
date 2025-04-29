import pandas as pd
from causal import plot_head_causal, plot_layer_causal
if __name__ == "__main__":
    print('==== Layer causal ====')
    with open("causal/layer_causal/results.csv", "r") as f:
        results = pd.read_csv(f)
    plot_layer_causal(results)
    
    print('==== Head causal ====')
    with open("causal/head_causal/results.csv", "r") as f:
        results = pd.read_csv(f)
    plot_head_causal(results)

    print('==== Answer Circuit ====')
    with open("circuit/answer_head.csv", "r") as f:
        df = pd.read_csv(f)
    df = df.groupby(['layer_id', 'head_id']).agg({"cos_true": "mean", "delta": "mean"}).reset_index()
    print("\n===== Top heads writing answer residual (cos_true desc) =====")
    cos_df = df.sort_values("cos_true", ascending=False).head(5).reset_index(drop=True)
    print(cos_df.head())
    print("\n===== Top heads writing answer residual (delta desc)=====")
    delta_df = df.sort_values("delta", ascending=False).head(5).reset_index(drop=True)
    print(delta_df.head())
    
    print('==== Compare Circuit ====')
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