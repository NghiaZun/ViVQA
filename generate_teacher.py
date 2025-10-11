"""
Day 1 - Generate Teacher Reasoning Answers using GPT-4o-mini
Author: <your-name>
"""

import os
import base64
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from time import sleep
from utils_prompt import SYSTEM_PROMPT, build_fewshot_prompt

# ====================================
# Config
# ====================================
CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
OUT_JSONL = "/kaggle/working/teacher_outputs_day1.jsonl"

MODEL_NAME = "gpt-4o-mini"
NUM_SAMPLES = 250  # khoảng 250 mẫu/ngày ~250k tokens

# ====================================
# Init OpenAI
# ====================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
assert client, "Missing OpenAI API Key"

# ====================================
# Load Dataset
# ====================================
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
subset = df.head(NUM_SAMPLES)
print(f"[INFO] Loaded {len(subset)} samples from ViVQA")

# ====================================
# Teacher Function
# ====================================
def call_teacher_gpt4o(image_path: str, question: str, retry=3) -> dict:
    """Call GPT-4o-mini to generate Answer + Reasoning in Vietnamese"""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    user_prompt = build_fewshot_prompt(question)

    for _ in range(retry):
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

            # parse “Answer:” và “Reasoning:” nếu có
            answer, reasoning = "", ""
            for line in content.splitlines():
                if line.lower().startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()
                elif line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()

            return {"answer": answer, "reasoning": reasoning, "raw": content}
        except Exception as e:
            print(f"[WARN] Error: {e}")
            sleep(2)
    return {"answer": "", "reasoning": "", "raw": ""}

# ====================================
# Main loop
# ====================================
results = []
for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Generating teacher answers"):
    image_id = str(row["img_id"])
    question = str(row["question"])
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        continue

    res = call_teacher_gpt4o(image_path, question)
    time.sleep(0.4)
    if res["answer"]:
        results.append({
            "img_id": image_id,
            "image_path": image_path,
            "question": question,
            "teacher_answer": res["answer"],
            "teacher_reasoning": res["reasoning"]
        })

    # auto-save mỗi 20 mẫu
    if len(results) % 20 == 0:
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Final save
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] Saved {len(results)} reasoning samples to {OUT_JSONL}")
