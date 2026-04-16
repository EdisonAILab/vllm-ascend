"""End-to-end inference batch-invariance test using transformers on Ascend NPU.

Verifies that the same prompt produces identical output tokens regardless
of batch size, using the MXFP8 model with greedy decoding (temperature=0).

Model: Qwen3-0.6B-MXFP8
Settings: greedy decoding (do_sample=False), max_new_tokens=64
"""
import os
import json
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/o00649568/b84411271/models/Qwen3-0.6B-MXFP8"
MAX_NEW_TOKENS = 64
DEVICE = "npu"

# GSM8K math questions
QUESTIONS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Kylar went to the store to get water and some snacks. He spent $3 on water and $6 on snacks. If he had $20 initially, how much money does he have left?",
    "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
    "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the rest of the time going at the same speed. How far is he from home at the end of those 4 hours?",
    "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
]

NUM_QUESTIONS = len(QUESTIONS)


def format_prompt(question):
    return "Solve step by step.\nQ: {}\nA:".format(question)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts):
    """Generate tokens for a batch of prompts with greedy decoding."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # greedy = temperature 0
        num_beams=1,
    )

    # Extract only the generated tokens (not the prompt)
    results = []
    for i in range(len(prompts)):
        prompt_len = input_ids[i].shape[0]
        generated_ids = outputs[i][prompt_len:].tolist()
        results.append(generated_ids)
    return results


def main():
    print("=" * 60)
    print("Inference Batch-Invariance Test (transformers)")
    print("Model: Qwen3-0.6B-MXFP8")
    print("Greedy decoding, max_new_tokens={}".format(MAX_NEW_TOKENS))
    print("=" * 60)

    prompts = [format_prompt(q) for q in QUESTIONS[:NUM_QUESTIONS]]

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map=DEVICE,
    )
    model.eval()
    print("Model loaded on {}\n".format(DEVICE))

    # ── Run 1: All prompts in one batch ──
    print("--- Run 1: Full batch ({} prompts) ---".format(NUM_QUESTIONS))
    full_tokens = generate_batch(model, tokenizer, prompts)
    for i, tokens in enumerate(full_tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True)[:60]
        print("  Q{}: {} tokens, '{}'".format(i, len(tokens), text.replace('\n', ' ')))

    # ── Run 2: One prompt at a time ──
    print("\n--- Run 2: One-by-one ---")
    single_tokens = []
    for prompt in prompts:
        tokens = generate_batch(model, tokenizer, [prompt])[0]
        single_tokens.append(tokens)

    # ── Run 3: Pairs ──
    print("\n--- Run 3: Pairs ---")
    pair_tokens = [None] * NUM_QUESTIONS
    for i in range(0, NUM_QUESTIONS, 2):
        batch = prompts[i:i + 2]
        results = generate_batch(model, tokenizer, batch)
        for j, tokens in enumerate(results):
            pair_tokens[i + j] = tokens

    # ── Compare ──
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    passed = 0
    failed = 0

    for i in range(NUM_QUESTIONS):
        full = full_tokens[i]
        single = single_tokens[i]
        pair = pair_tokens[i]

        full_vs_single = (full == single)
        full_vs_pair = (full == pair)

        if full_vs_single and full_vs_pair:
            print("  Q{}: MATCH ({} tokens)".format(i, len(full)))
            passed += 1
        else:
            failed += 1
            if not full_vs_single:
                min_len = min(len(full), len(single))
                div = next((k for k in range(min_len) if full[k] != single[k]), min_len)
                print("  Q{}: FAIL full vs single diverge at token {}".format(i, div))
            if not full_vs_pair:
                min_len = min(len(full), len(pair))
                div = next((k for k in range(min_len) if full[k] != pair[k]), min_len)
                print("  Q{}: FAIL full vs pair diverge at token {}".format(i, div))

    print("\n" + "=" * 60)
    status = "ALL PASSED" if failed == 0 else "{} FAILED".format(failed)
    print("  Results: {}/{} - {}".format(passed, passed + failed, status))
    print("  Batch-invariant: {}".format("YES" if failed == 0 else "NO"))
    print("=" * 60)

    results = {
        "full_batch": {str(i): t for i, t in enumerate(full_tokens)},
        "single": {str(i): t for i, t in enumerate(single_tokens)},
        "pairs": {str(i): t for i, t in enumerate(pair_tokens)},
        "passed": passed,
        "failed": failed,
    }
    with open("/home/o00649568/b84411271/bi_inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to bi_inference_results.json")


if __name__ == "__main__":
    main()
