# Finding Year Comparison Circuit in LLMs

[Report](https://github.com/WitchPuff/MI-YearCircuit/blob/main/report.md)

## Environment

```python
conda env create -f environment.yml
conda activate mi
```

## Run

To print all results simply:

```python
python print_all_results.py
```

And see all visualisation in `causal/`, `circuit/`, `probing/`. More structured and emphasized tables and figures can be seen in the [report](https://github.com/WitchPuff/MI-YearCircuit/blob/main/report.pdf). 

To replicate the experiments:

#### 1) Data

```shell
# To check how the model tokenize a 4-digit year.
python token_count.py 
# To generate examples of prompts
python data.py
# To check the architecture of model microsoft/Phi-3-mini-4k-instruct
python model.py
# Build dataset: dataset_qa_yesno_150_1800_2020.csv
python build_dataset.py \
    --n_samples 150 \
    --max_year 2020 \
    --min_year 1800 \
    --prompt_style qa_yesno \
    --fewshot 0 \
    --condition \ # this will make sure the first digits differ in two years
    --verbose

```

#### 2) Experiments

```shell
# The year range below should be same as in the csv path.
# Probing
python probing.py --min_year 1000 --max_year 2020 # The raw results are stored in probing/1000_2020/results_probing.json

python probing.py --min_year 1800 --max_year 2020 # The raw results are stored in probing/1800_2020/results_probing.json

# causal
# The raw results are stored in causal/layer_causal/results.csv and causal/head_causal/results.csv
python causal.py --min_year 1800 --max_year 2020 --sample_size 10

# compare head
# The raw results are stored in circuit/1800_2020_10/compare_circuit.csv
python compare_head.py --min_year 1800 --max_year 2020 --sample_size 10 

# The raw results are stored in circuit/1800_2020_10_differ_in_first_digit/compare_circuit.csv
python compare_head.py --min_year 1800 --max_year 2020 --sample_size 10 --condition # this will make sure the first digits differ in two years

# answer head
# The raw results are stored in circuit/1800_2020_10/answer_head.csv
python answer_head.py --min_year 1800 --max_year 2020 --sample_size 10
```

 