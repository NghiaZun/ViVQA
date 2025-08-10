import pandas as pd
from evaluate import load

# Danh sách đường dẫn các file CSV
files = [
    "vqa_predictions.csv",
    "vqa_predictions (1).csv",
    "vqa_predictions (2).csv",
    "vqa_predictions (3).csv"
]

# Gộp lại
dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)

# Chuẩn hóa tên cột nếu cần
df_all.columns = [col.strip().lower() for col in df_all.columns]
df_all = df_all.rename(columns={"gt": "gt_answer"})

# Tính Exact Match
df_all["em"] = (df_all["gt_answer"].str.strip() == df_all["pred_answer"].str.strip()).astype(int)
print(f"Exact Match (EM): {df_all['em'].mean():.2%}")

# Tính BLEU và ROUGE
bleu = load("bleu")
rouge = load("rouge")

preds = df_all["pred_answer"].astype(str).tolist()
refs = df_all["gt_answer"].astype(str).tolist()

# BLEU expects list of list for references
bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
rouge_score = rouge.compute(predictions=preds, references=refs)

print("BLEU:", bleu_score)
print("ROUGE:", rouge_score)