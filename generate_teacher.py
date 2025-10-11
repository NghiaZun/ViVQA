"""
Day 1 - Generate Teacher Reasoning Answers using GPT-4o-mini (Stable version)
Author: Nghia Duong
Description:
- Supports auto resume
- Smart rate limiter (avoid 429)
- Generates Vietnamese reasoning answers for ViVQA
"""

import os
import base64
import time
import json
import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from utils_prompt import SYSTEM_PROMPT, build_fewshot_prompt

# ======================================================
# Config
# ======================================================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
OUT_JSONL = "/kaggle/working/teacher_outputs_day1.jsonl"

MODEL_NAME = "gpt-4o-mini"
NUM_SAMPLES = 150          # mỗi ngày sinh ~150 mẫu
TOKENS_PER_REQUEST = 1000  # ước lượng
MAX_TOKENS_PER_MIN = 180000  # giữ an toàn <200k
REQUEST_SLEEP = 1.2         # giây giữa các request

# ======================================================
# Init OpenAI
# ======================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
assert client, "❌ Missing OPENAI_API_KEY in Kaggle Secrets!"

# ======================================================
# Resume previous results (if exist)
# ======================================================
results = []
if os.path.exists(OUT_JSONL):
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
done_ids = {r["img_id"] for r in results}
print(f"[INFO] Resume mode: found {len(done_ids)} completed samples")

# ======================================================
# Load Dataset
# ======================================================
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
subset = df.head(NUM_SAMPLES)
print(f"[INFO] Loaded {len(subset)} samples from ViVQA")

# ======================================================
# Teacher Call Function (with retry + backoff)
# ======================================================
def call_teacher_gpt4o(image_path: str, question: str, retry=5) -> dict:
    """Call GPT-4o-mini to generate Vietnamese Answer + Reasoning"""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    user_prompt = build_fewshot_prompt(question)
    backoff = 2  # giây

    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.4,
            )

            content = response.choices[0].message.content.strip()

            answer, reasoning = "", ""
            for line in content.splitlines():
                if line.lower().startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()
                elif line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()

            return {"answer": answer, "reasoning": reasoning, "raw": content}

        except Exception as e:
            err_msg = str(e)
            print(f"[WARN] Error: {err_msg}")
            if "rate_limit" in err_msg or "429" in err_msg:
                sleep_time = backoff + random.uniform(0, 1)
                print(f"[INFO] 💤 Rate limit hit — sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                backoff = min(backoff * 2, 60)  # exponential backoff
            else:
                time.sleep(2)
    return {"answer": "", "reasoning": "", "raw": ""}


# ======================================================
# Rate Limiter (global TPM control)
# ======================================================
tokens_this_minute = 0
minute_start = time.time()

def check_rate_limit():
    global tokens_this_minute, minute_start
    elapsed = time.time() - minute_start
    if elapsed < 60 and tokens_this_minute > MAX_TOKENS_PER_MIN:
        wait_time = 60 - elapsed
        print(f"[INFO] 🧭 Waiting {wait_time:.1f}s to reset TPM window...")
        time.sleep(wait_time)
        tokens_this_minute = 0
        minute_start = time.time()
    elif elapsed >= 60:
        tokens_this_minute = 0
        minute_start = time.time()


# ======================================================
# Main Loop
# ======================================================
for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Generating teacher answers"):
    image_id = str(row["img_id"])
    question = str(row["question"])
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

    if image_id in done_ids:
        continue
    if not os.path.exists(image_path):
        continue

    check_rate_limit()
    res = call_teacher_gpt4o(image_path, question)
    tokens_this_minute += TOKENS_PER_REQUEST
    time.sleep(REQUEST_SLEEP)

    if res["answer"]:
        results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": question,
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"]
        })

        # Append ngay vào file (an toàn)
        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")

    if len(results) % 20 == 0:
        print(f"[CHECKPOINT] Saved {len(results)} samples so far")

print(f"[INFO] ✅ Finished — total {len(results)} reasoning samples saved to {OUT_JSONL}")
