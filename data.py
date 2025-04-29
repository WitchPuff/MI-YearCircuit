import random
import pandas as pd
import os


def get_year_pair(max_year=2020, min_year=1950):
    year1 = random.randint(min_year, max_year)
    year2 = random.randint(min_year, max_year)
    while year1 == year2:
        year2 = random.randint(min_year, max_year)
    return year1, year2

def get_prompt(max_year=2020, min_year=1950, prompt_style='qa_yesno', example_num=0):

    year1, year2 = get_year_pair(max_year, min_year)
    prompt = "<system> You are a bot that can compare years. <system>\n"
    true_label = "True"
    false_label = "False"

    if prompt_style == "qa":
        for _ in range(example_num):
            year3, year4 = get_year_pair(max_year, min_year)
            if year1 == year3 or year2 == year4:
                continue
            prompt += f"Q: The year {year3} is earlier than {year4}. True or False?\nA: {true_label if year3 < year4 else false_label}\n"
        prompt += f"Q: The year {year1} is earlier than {year2}. True or False?\nA:"

    elif prompt_style == "tf":
        for _ in range(example_num):
            year3, year4 = get_year_pair(max_year, min_year)
            if year1 == year3 or year2 == year4:
                continue
            prompt += f"The year {year3} is earlier than {year4}. That is {true_label if year3 < year4 else false_label}\n"
        prompt += f"The year {year1} is earlier than {year2}. That is"

    elif prompt_style == "qa_yesno":
        true_label = "Yes"
        false_label = "No"
        for _ in range(example_num):
            year3, year4 = get_year_pair(max_year, min_year)
            if year1 == year3 or year2 == year4:
                continue
            prompt += f"Q: Is the year {year3} earlier than {year4}? Answer with 'Yes' or 'No'. \nA: {true_label if year3 < year4 else false_label}\n"
        prompt += f"Please answer: Is {year1} earlier than {year2}? Answer with 'Yes' or 'No'.\nA:"
    
    # elif prompt_style == "year":
    #     prompt = f"Between {year1} and {year2}, the earlier year is {year1 if year1 < year2 else year2}\n"
    #     prompt = f"Between {year3} and {year4}, the earlier year is {year3 if year3 < year4 else year4}\n"
    #     prompt = f"Between {year5} and {year6}, the earlier year is"
    #     true_token_id = tokenizer.encode(str(year5), add_special_tokens=False)[0]
    #     false_token_id = tokenizer.encode(str(year6), add_special_tokens=False)[0]
    #     label = " True" if year5 < year6 else " False"
    else:
        raise ValueError("Unsupported prompt_style")
    
    label = f" {true_label}" if year1 < year2 else f" {false_label}"

    return prompt, label, (true_label, false_label), (year1, year2)


def load_data(csv_path, sample_size=None):
    df = pd.read_csv(csv_path)
    
    if sample_size is not None:
        if os.path.exists(f"{csv_path[:-4]}_{sample_size}.csv"):
            df = pd.read_csv(f"{csv_path[:-4]}_{sample_size}.csv")
            print(f"Loaded {csv_path[:-4]}_{sample_size}.csv")
        else:
            num_labels = df['label'].nunique()
            samples_per_label = sample_size // num_labels

            df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(samples_per_label, len(x)), random_state=42))
            df.to_csv(f"{csv_path[:-4]}_{sample_size}.csv", index=False)
        
    prompts = df['prompt'].tolist()
    labels = df['label'].tolist()
    return prompts, labels

if __name__ == "__main__":
    n_samples = 100
    max_year = 2025
    min_year = 1000
    prompt_style = "qa_yesno"
    example_num = 0
    for i in range(n_samples):
        prompt, label, (true_label, false_label) = get_prompt(max_year=max_year, min_year=min_year, prompt_style=prompt_style, example_num=example_num)
        print(f"Prompt {i+1}:", prompt)
        print("Label:", label)
        print("True Label:", true_label)
        print("False Label:", false_label)