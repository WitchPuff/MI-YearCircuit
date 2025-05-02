import pandas as pd
from causal import plot_head_causal, plot_layer_causal
from answer_head import visualize_answer_heads
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
    
    visualize_answer_heads(min_year=1800, max_year=2020, sample_size=10, condition=False, ret_dir='circuit')
    # visualize_answer_heads(min_year=1800, max_year=2020, sample_size=10, condition=True, ret_dir=None)
    
    
    
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