import pandas as pd
from causal import plot_head_causal, plot_layer_causal
from answer_head import plot_answer_head, plot_attn_tokens, count_attn_tokens
from collections import Counter
import pandas as pd




if __name__ == "__main__":
    # print('==== Layer causal ====')
    # with open("causal/layer_causal/results.csv", "r") as f:
    #     results = pd.read_csv(f)
    # plot_layer_causal(results)
    
    # print('==== Head causal ====')
    # with open("causal/head_causal/results.csv", "r") as f:
    #     results = pd.read_csv(f)
    # plot_head_causal(results)

    print('==== Answer Circuit ====')
    with open("circuit/answer_head.csv", "r") as f:
        df = pd.read_csv(f)
    attn_freq_df = count_attn_tokens(df)
    df = df.groupby(['layer_id', 'head_id']).agg({"cos_true": "mean", "delta": "mean"}).reset_index()
    merged_df = pd.merge(attn_freq_df, df, on=["layer_id", "head_id"], how="left")
    plot_attn_tokens(merged_df)
    plot_answer_head(df, metrics=["delta", "cos_true"])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("\n===== Top heads writing answer residual (cos_true desc) =====")
    cos_df = df.sort_values("cos_true", ascending=False).head(15).reset_index(drop=True)
    print(cos_df)
    merged_df = pd.merge(attn_freq_df, cos_df, on=["layer_id", "head_id"], how="right")
    merged_df = merged_df.sort_values("cos_true", ascending=False)
    print(merged_df)
    print("\n===== Top heads writing answer residual (delta desc)=====")
    delta_df = df.sort_values("delta", ascending=False).head(15).reset_index(drop=True)
    print(delta_df)
    merged_df = pd.merge(attn_freq_df, delta_df, on=["layer_id", "head_id"], how="right")
    merged_df = merged_df.sort_values("delta", ascending=False)
    print(merged_df)
    # print('==== Compare Circuit ====')
    # with open("circuit/compare_circuit.csv", "r") as f:
    #     df = pd.read_csv(f)
    # df = df.groupby(['layer_id', 'head_id', 'bit']).agg({"score_b2a": "mean",
    #                                               "sim_a": "mean",
    #                                               "sim_b": "mean"}).reset_index()
    # for bit in range(4):
    #     bit_df = df[df['bit'] == bit]
    #     for metric in ['score_b2a', 'sim_a', 'sim_b']:
    #         print(f"\n===== Top heads compare heads (Bit{bit}, {metric} desc) =====")
    #         ret_df = bit_df.sort_values(metric, ascending=False if metric is not "sim_b" else F).head(5).reset_index(drop=True)
    #         print(ret_df.head())