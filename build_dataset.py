from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer
from data import get_prompt
import pandas as pd





def sanity_check(
    model,
    tokenizer,
    n_samples=100,
    min_year=1950,
    max_year=2020,
    prompt_style="qa_yesno", # this works for phi3
    example_num=3,
    use_logits=False,
    verbose=False,
    save_dataset=True
):
    correct = 0
    total = 0
    results = []
    
    for i in tqdm(range(n_samples)):

        prompt, label, (true_label, false_label), (year1, year2) = get_prompt(
            max_year, min_year, prompt_style=prompt_style, example_num=example_num
        )
        true_token_id = tokenizer.encode(true_label, add_special_tokens=False)[0]
        false_token_id = tokenizer.encode(false_label, add_special_tokens=False)[0]


        tokens = model.to_tokens(prompt)
        print(tokens.shape)
        if use_logits:
            logits = model(tokens)
            final_logits = logits[0, -1]
            probs = torch.softmax(final_logits, dim=-1)
            true_logit = final_logits[true_token_id].item()
            false_logit = final_logits[false_token_id].item()

            pred_str = true_label if true_logit > false_logit else false_label

            is_correct = (pred_str.strip() == label.strip())
            correct += is_correct
            total += 1
            
        results.append({
            "label": label.strip(),
            "prediction": pred_str.strip(),
            "is_correct": is_correct,
            "prompt": prompt,
            "tokens_len": tokens.shape[1],
            "year1": year1,
            "year2": year2,
            
        })


        if verbose:
            print(f"{prompt} | Prediction: '{pred_str.strip()}' | Ground Truth: '{label.strip()}' | {'Correct' if is_correct else 'Wrong'}")
            print(f"P({true_label}):", probs[true_token_id].item(), f"| P({false_label}):", probs[false_token_id].item())

    accuracy = correct / total
    print(f"Accuracy on Year Comparison Task (Prompt style: {prompt_style}): {accuracy*100:.2f}%")

    if save_dataset:
        dataset = pd.DataFrame([r for r in results if r["is_correct"]])
        dataset = dataset.drop(columns=["is_correct", "prediction"])
        print(dataset.head())
        dataset.to_csv(f"dataset_{prompt_style}_{n_samples}.csv", index=False)
        print(f"Dataset saved to dataset_{prompt_style}_{n_samples}.csv")
    
    return accuracy, results
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    # model  = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model = model.to('cuda')
    acc, results = sanity_check(
        model=model,
        tokenizer=tokenizer,
        n_samples=150,
        max_year=2020,
        min_year=1800,
        prompt_style="qa_yesno",
        example_num=0,
        use_logits=True,
        verbose=True
    )
