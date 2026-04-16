"""End-to-end inference batch-invariance test using vLLM on Ascend NPU.

Verifies that the same prompt produces identical output tokens regardless
of how many other prompts are in the same batch.

Model: Qwen3-0.6B-MXFP8
Dataset: GSM8K (subset)
Settings: temperature=0, tensor_parallel=1
"""
import os
import json

# Must set before any torch import
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams


MODEL_PATH = "/home/o00649568/b84411271/models/Qwen3-0.6B-MXFP8"
MAX_TOKENS = 128
NUM_QUESTIONS = 10

# GSM8K-style math questions (no dataset download needed)
GSM8K_QUESTIONS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Every day, Wendi feeds each of her 6 chickens 3 cups of mixed chicken feed, containing seeds, mealworms and vegetables to keep them healthy. She gives the chickens their feed in three equal meals. If a meal is 1 cup of feed, how many cups of feed does she need for all the chickens for one meal?",
    "Kylar went to the store to get water and some snacks. He spent $3 on water and $6 on snacks. If he had $20 initially, how much money does he have left?",
    "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
    "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?",
    "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the rest of the time going at the same speed. How far is he from home at the end of those 4 hours?",
    "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
]


def format_prompt(question):
    return "Solve the following math problem step by step.\n\nQuestion: {}\n\nAnswer:".format(question)


def main():
    print("=" * 60)
    print("Inference Batch-Invariance Test")
    print("Model: Qwen3-0.6B-MXFP8")
    print("temperature=0, tp=1, VLLM_BATCH_INVARIANT=1")
    print("=" * 60)

    prompts = [format_prompt(q) for q in GSM8K_QUESTIONS[:NUM_QUESTIONS]]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
    )

    print("\nLoading model...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        dtype="auto",
        trust_remote_code=True,
        enforce_eager=True,
    )
    print("Model loaded.\n")

    # ── Run 1: All prompts in one batch ──
    print("--- Run 1: Full batch (all {} prompts) ---".format(NUM_QUESTIONS))
    outputs_full = llm.generate(prompts, sampling_params)
    full_tokens = {}
    for i, output in enumerate(outputs_full):
        tokens = output.outputs[0].token_ids
        text = output.outputs[0].text
        full_tokens[i] = list(tokens)
        print("  Q{}: {} tokens, starts: '{}'".format(
            i, len(tokens), text[:60].replace('\n', ' ')))

    # ── Run 2: One prompt at a time ──
    print("\n--- Run 2: One-by-one (batch_size=1) ---")
    single_tokens = {}
    for i, prompt in enumerate(prompts):
        output = llm.generate([prompt], sampling_params)[0]
        tokens = output.outputs[0].token_ids
        single_tokens[i] = list(tokens)

    # ── Run 3: Pairs ──
    print("\n--- Run 3: Pairs (batch_size=2) ---")
    pair_tokens = {}
    for i in range(0, NUM_QUESTIONS, 2):
        batch = prompts[i:i+2]
        outputs = llm.generate(batch, sampling_params)
        for j, output in enumerate(outputs):
            tokens = output.outputs[0].token_ids
            pair_tokens[i + j] = list(tokens)

    # ── Compare results ──
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
            print("  Q{}: MATCH (all 3 runs identical, {} tokens)".format(i, len(full)))
            passed += 1
        else:
            failed += 1
            if not full_vs_single:
                # Find first divergence
                min_len = min(len(full), len(single))
                div_pos = min_len
                for k in range(min_len):
                    if full[k] != single[k]:
                        div_pos = k
                        break
                print("  Q{}: FAIL full vs single diverge at token {} "
                      "(full={}, single={})".format(
                          i, div_pos,
                          full[div_pos] if div_pos < len(full) else "END",
                          single[div_pos] if div_pos < len(single) else "END"))
            if not full_vs_pair:
                min_len = min(len(full), len(pair))
                div_pos = min_len
                for k in range(min_len):
                    if full[k] != pair[k]:
                        div_pos = k
                        break
                print("  Q{}: FAIL full vs pair diverge at token {}".format(i, div_pos))

    print("\n" + "=" * 60)
    total = passed + failed
    status = "ALL PASSED" if failed == 0 else "{} FAILED".format(failed)
    print("  Results: {}/{} questions - {}".format(passed, total, status))
    print("  Batch-invariant: {}".format("YES" if failed == 0 else "NO"))
    print("=" * 60)

    # Save results for reference
    results = {
        "full_batch": full_tokens,
        "single": single_tokens,
        "pairs": pair_tokens,
        "passed": passed,
        "failed": failed,
    }
    with open("/home/o00649568/b84411271/bi_inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to bi_inference_results.json")


if __name__ == "__main__":
    main()
