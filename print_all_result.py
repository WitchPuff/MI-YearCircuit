import pandas as pd
from causal import plot_head_causal, plot_layer_causal
from compare_head import visualize_compare_head
from answer_head import visualize_answer_heads




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
    visualize_answer_heads(min_year=1800, max_year=2020, sample_size=10, condition=False, ret_dir=None)
    # visualize_answer_heads(min_year=1800, max_year=2020, sample_size=10, condition=True, ret_dir=None)
    
    
    
    print('==== Compare Circuit ====')
    visualize_compare_head(min_year=1800, max_year=2020, sample_size=10, condition=False, ret_dir=None)
    visualize_compare_head(min_year=1800, max_year=2020, sample_size=10, condition=True, ret_dir=None)