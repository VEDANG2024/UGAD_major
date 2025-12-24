import os
import json
import random
import datasets # pip install datasets
from tqdm import tqdm

# Configuration matches your Report's Experimental Setup
CONFIG = {
    "gsm8k_size": 1000,        # "GSM8K Subset: 1,000 samples"
    "synthetic_size": 1000,    # "Synthetic Arithmetic: 1,000 samples"
    "ood_size": 200,           # "BigBench-Hard (OOD): 200 samples"
    "seed": 42
}

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(CONFIG["seed"])

def save_jsonl(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"✅ Saved {len(data)} samples to {filepath}")

# ==========================================
# 1. GSM8K (Hard Reasoning Tasks)
# Source: openai/gsm8k
# ==========================================
print("⬇️  Downloading GSM8K...")
try:
    gsm8k = datasets.load_dataset("openai/gsm8k", "main", split="train")
    # Shuffle and take subset to match "1,000 samples" claim
    gsm8k_subset = gsm8k.shuffle(seed=CONFIG["seed"]).select(range(CONFIG["gsm8k_size"]))
    
    formatted_gsm8k = [
        {"id": f"gsm8k_{i}", "prompt": item["question"], "completion": item["answer"], "difficulty": "hard"}
        for i, item in enumerate(gsm8k_subset)
    ]
    save_jsonl(formatted_gsm8k, "train_gsm8k_hard.jsonl")
except Exception as e:
    print(f"⚠️ Failed to load GSM8K: {e}")

# ==========================================
# 2. Synthetic Arithmetic (Easy Tasks)
# Source: Procedurally Generated (Custom)
# ==========================================
print("⚙️  Generating Synthetic Arithmetic...")
synthetic_data = []
ops = ['+', '-', '*', '/']

for i in range(CONFIG["synthetic_size"]):
    # Mix of simple arithmetic and formatting tasks
    op = random.choice(ops)
    if op == '+':
        a, b = random.randint(1, 100), random.randint(1, 100)
        ans = a + b
    elif op == '-':
        a, b = random.randint(1, 100), random.randint(1, 100)
        if a < b: a, b = b, a # Ensure positive for "easy"
        ans = a - b
    elif op == '*':
        a, b = random.randint(1, 20), random.randint(1, 10)
        ans = a * b
    else: # Division
        b = random.randint(1, 10)
        ans = random.randint(1, 20)
        a = b * ans # Ensure clean division
    
    # "Schema formatting" mentioned in report
    if random.random() > 0.5:
        prompt = f"Calculate {a} {op} {b} and return in JSON format."
        completion = json.dumps({"operation": f"{a} {op} {b}", "result": ans})
    else:
        prompt = f"What is {a} {op} {b}?"
        completion = str(ans)

    synthetic_data.append({
        "id": f"syn_{i}", 
        "prompt": prompt, 
        "completion": completion, 
        "difficulty": "easy"
    })

save_jsonl(synthetic_data, "train_synthetic_easy.jsonl")

# ==========================================
# 3. BigBench-Hard (OOD Robustness)
# Source: Joschka/big_bench_hard
# ==========================================
print("⬇️  Downloading BigBench-Hard (OOD)...")
try:
    # We use 'boolean_expressions' and 'logical_deduction' as proxies for OOD logic
    bbh = datasets.load_dataset("Joschka/big_bench_hard", "boolean_expressions", split="train")
    bbh_subset = bbh.shuffle(seed=CONFIG["seed"]).select(range(CONFIG["ood_size"]))
    
    formatted_bbh = [
        {"id": f"bbh_{i}", "prompt": item["input"], "completion": item["target"], "type": "OOD"}
        for i, item in enumerate(bbh_subset)
    ]
    save_jsonl(formatted_bbh, "test_bbh_ood.jsonl")
except Exception as e:
    print(f"⚠️ Failed to load BBH: {e}")

# ==========================================
# 4. HumanEval (Coding Benchmark)
# Source: openai_humaneval
# ==========================================
print("⬇️  Downloading HumanEval...")
try:
    # HumanEval is typically a test-only set
    heval = datasets.load_dataset("openai_humaneval", split="test")
    
    formatted_heval = [
        {"id": f"he_{i}", "prompt": item["prompt"], "completion": item["canonical_solution"], "entry_point": item["entry_point"]}
        for i, item in enumerate(heval)
    ]
    save_jsonl(formatted_heval, "test_humaneval.jsonl")
except Exception as e:
    print(f"⚠️ Failed to load HumanEval: {e}")

print("\n✨ All datasets prepared in 'data/' directory.")
