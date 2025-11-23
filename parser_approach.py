"""
POST-PROCESSING PARSER APPROACH
Model sinh text tự do, parser extract answer + reasoning

Ưu điểm:
- Đơn giản hơn, model tập trung vào nội dung
- 100% parse được
- Dễ debug và cải thiện parser
"""

import re
from typing import Dict, Tuple

# ============================================================================
# PARSER STRATEGIES
# ============================================================================

def parse_v1_simple(text: str) -> Dict[str, str]:
    """
    Strategy 1: Simple heuristic
    Giả định: Answer ở đầu, reasoning ở sau
    """
    lines = text.strip().split('\n')
    
    # First non-empty line = answer
    answer = ""
    reasoning = ""
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if not answer:
            answer = line
        else:
            reasoning += line + " "
    
    return {
        "answer": answer.strip(),
        "reasoning": reasoning.strip()
    }


def parse_v2_keywords(text: str) -> Dict[str, str]:
    """
    Strategy 2: Keyword-based
    Tìm keywords như "Trả lời:", "Lý do:", "Giải thích:"
    """
    text = text.strip()
    
    # Common Vietnamese keywords
    answer_keywords = [
        "trả lời:", "đáp án:", "câu trả lời:",
        "answer:", "result:"
    ]
    reasoning_keywords = [
        "lý do:", "giải thích:", "vì:", "bởi vì:",
        "reasoning:", "because:", "explanation:"
    ]
    
    answer = ""
    reasoning = ""
    
    # Try to find answer section
    for kw in answer_keywords:
        if kw in text.lower():
            parts = text.lower().split(kw, 1)
            if len(parts) == 2:
                # Get text after keyword until next keyword or end
                after = parts[1]
                # Stop at reasoning keyword
                for rkw in reasoning_keywords:
                    if rkw in after:
                        answer = after.split(rkw)[0].strip()
                        break
                else:
                    # No reasoning keyword, take first sentence
                    answer = after.split('.')[0].strip()
                break
    
    # Try to find reasoning section
    for kw in reasoning_keywords:
        if kw in text.lower():
            parts = text.lower().split(kw, 1)
            if len(parts) == 2:
                reasoning = parts[1].strip()
                break
    
    # Fallback: use simple strategy
    if not answer:
        result = parse_v1_simple(text)
        answer = result["answer"]
        reasoning = result["reasoning"]
    
    return {
        "answer": answer,
        "reasoning": reasoning
    }


def parse_v3_ml_based(text: str) -> Dict[str, str]:
    """
    Strategy 3: ML-based extraction
    Train một classifier nhỏ để phân loại câu nào là answer, câu nào là reasoning
    """
    # Placeholder - có thể dùng:
    # - Sentence-BERT để classify
    # - PhoBERT fine-tuned trên answer/reasoning classification
    # - Simple features: position, length, POS tags
    
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return {"answer": "", "reasoning": ""}
    
    # Heuristic: Short first sentence = answer, rest = reasoning
    answer = sentences[0]
    reasoning = " ".join(sentences[1:])
    
    return {
        "answer": answer,
        "reasoning": reasoning
    }


def parse_v4_template_based(text: str) -> Dict[str, str]:
    """
    Strategy 4: Template matching
    Model được train để output theo template có thể parse
    Ví dụ: "Answer is X. This is because Y."
    """
    # Pattern 1: "Answer is X. Because/Since Y."
    pattern1 = r"(?:answer is|đáp án là)\s+(.+?)\.\s*(?:because|since|vì|do)\s+(.+)"
    match1 = re.search(pattern1, text, re.IGNORECASE | re.DOTALL)
    if match1:
        return {
            "answer": match1.group(1).strip(),
            "reasoning": match1.group(2).strip()
        }
    
    # Pattern 2: "X. Y." (first sentence answer, rest reasoning)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        return {
            "answer": sentences[0].rstrip('.!?'),
            "reasoning": " ".join(sentences[1:])
        }
    
    return {"answer": text, "reasoning": ""}


def parse_v5_ensemble(text: str) -> Dict[str, str]:
    """
    Strategy 5: Ensemble of parsers
    Chạy nhiều parsers, vote/merge kết quả
    """
    results = [
        parse_v1_simple(text),
        parse_v2_keywords(text),
        parse_v4_template_based(text)
    ]
    
    # Simple voting: pick most common answer
    answers = [r["answer"] for r in results if r["answer"]]
    reasonings = [r["reasoning"] for r in results if r["reasoning"]]
    
    # Pick longest (usually most complete)
    answer = max(answers, key=len) if answers else ""
    reasoning = max(reasonings, key=len) if reasonings else ""
    
    return {
        "answer": answer,
        "reasoning": reasoning
    }


# ============================================================================
# TRAINING MODIFICATION
# ============================================================================

def prepare_natural_training_data():
    """
    Thay vì train với XML:
      <answer>X</answer><reasoning>[TYPE] Y</reasoning>
    
    Train với natural text:
      X. Y.
    hoặc:
      Answer: X. Reasoning: Y.
    hoặc:
      X because Y.
    """
    template_styles = [
        # Style 1: Direct
        "{answer}. {reasoning}.",
        
        # Style 2: Explicit labels
        "Answer: {answer}. Reasoning: {reasoning}.",
        
        # Style 3: Natural flow
        "{answer} because {reasoning}.",
        
        # Style 4: Vietnamese natural
        "Đáp án là {answer}. Lý do là {reasoning}.",
        
        # Style 5: Explanation first
        "{reasoning}. Therefore, the answer is {answer}."
    ]
    
    # Randomly pick style during training for diversity
    import random
    return random.choice(template_styles)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_parser(predictions, ground_truth):
    """
    Test parser accuracy
    """
    correct_answers = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        parsed = parse_v2_keywords(pred)  # or any parser
        
        if parsed["answer"].lower().strip() == gt["answer"].lower().strip():
            correct_answers += 1
    
    accuracy = correct_answers / total * 100
    print(f"Parser Accuracy: {accuracy:.2f}%")
    return accuracy


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Case 1: Natural text
        "Có 3 người. Vì trong ảnh có 3 người đang đứng.",
        
        # Case 2: With keywords
        "Trả lời: Có 3 người. Lý do: Trong ảnh có 3 người đang đứng.",
        
        # Case 3: English style
        "Answer: 3 people. Because there are 3 people standing in the image.",
        
        # Case 4: Explanation first
        "Trong ảnh có 3 người đang đứng ở công viên. Do đó đáp án là 3 người.",
        
        # Case 5: No clear structure
        "Có 3 người đang đứng nên câu trả lời là 3 người."
    ]
    
    print("="*70)
    print("TESTING DIFFERENT PARSERS")
    print("="*70)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Input:")
        print(f"  {text}")
        print(f"\nParsed results:")
        
        # Test each parser
        for parser_name, parser_func in [
            ("Simple", parse_v1_simple),
            ("Keyword", parse_v2_keywords),
            ("Template", parse_v4_template_based),
            ("Ensemble", parse_v5_ensemble)
        ]:
            result = parser_func(text)
            print(f"\n  [{parser_name}]")
            print(f"    Answer:    {result['answer'][:50]}...")
            print(f"    Reasoning: {result['reasoning'][:50]}...")
