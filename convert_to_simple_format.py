"""
Convert teacher_outputs.jsonl to Simple Format
Transform XML-based data to natural Answer:/Reasoning: format
"""

import json
from tqdm import tqdm

# Paths
INPUT_JSONL = "kaggle/teacher_outputs.jsonl"
OUTPUT_JSONL = "kaggle/teacher_outputs_simple.jsonl"

def convert_to_simple_format(input_file, output_file):
    """
    Convert from XML format to simple format
    
    Input fields:
    - teacher_answer: "Đen"
    - teacher_reasoning: "Màu sắc của chiếc váy trong hình ảnh là đen."
    - teacher_raw: "<answer>Đen</answer>\n<reasoning>[TYPE] ..."
    
    Output:
    - Add: teacher_simple: "Answer: Đen\nReasoning: Màu sắc..."
    - Keep all other fields for compatibility
    """
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Converting"):
            item = json.loads(line)
            
            # Extract answer and reasoning
            answer = item.get("teacher_answer", "").strip()
            reasoning = item.get("teacher_reasoning", "").strip()
            
            # Skip if missing data
            if not answer or not reasoning:
                continue
            
            # Create simple format
            teacher_simple = f"Answer: {answer}\nReasoning: {reasoning}"
            
            # Add to item
            item["teacher_simple"] = teacher_simple
            
            # Write to output
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            converted_count += 1
    
    return converted_count

if __name__ == "__main__":
    print("="*70)
    print("CONVERTING TO SIMPLE FORMAT")
    print("="*70)
    print(f"Input:  {INPUT_JSONL}")
    print(f"Output: {OUTPUT_JSONL}")
    print()
    
    count = convert_to_simple_format(INPUT_JSONL, OUTPUT_JSONL)
    
    print()
    print("="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Converted: {count} samples")
    print(f"Saved to: {OUTPUT_JSONL}")
    print()
    
    # Show sample
    print("Sample output:")
    with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
        print(f"\nQuestion: {sample['question']}")
        print(f"\nSimple format:")
        print(sample['teacher_simple'])
        print()
        print(f"Original XML:")
        print(sample.get('teacher_raw', 'N/A')[:100] + '...')
    
    print("="*70)
