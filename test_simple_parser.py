"""
Test Simple Format Parser
Verify that parser can extract Answer and Reasoning reliably
"""

import re

def parse_output(text: str):
    """
    Parse output format:
    Answer: ...
    Reasoning: ...
    """
    answer = ""
    reasoning = ""
    
    # Try regex first
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Fallback: line-based parsing
    if not answer or not reasoning:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip()
            elif line.lower().startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid': bool(answer and reasoning)
    }


# Test cases
test_cases = [
    # Perfect format
    "Answer: Đen\nReasoning: Màu sắc của chiếc váy trong hình ảnh là đen.",
    
    # With extra text before
    "Dựa vào ảnh:\nAnswer: Đen\nReasoning: Màu sắc của chiếc váy trong hình ảnh là đen.",
    
    # No newline
    "Answer: Đen Reasoning: Màu sắc của chiếc váy trong hình ảnh là đen.",
    
    # Case insensitive
    "answer: Đen\nreasoning: Màu sắc của chiếc váy trong hình ảnh là đen.",
    
    # Multi-line reasoning
    "Answer: Đen\nReasoning: Màu sắc của chiếc váy trong hình ảnh là đen.\nVáy có màu đen tuyền.",
    
    # Extra spaces
    "Answer:  Đen  \nReasoning:  Màu sắc của chiếc váy trong hình ảnh là đen.  ",
]

print("="*70)
print("TESTING SIMPLE FORMAT PARSER")
print("="*70)

for i, text in enumerate(test_cases, 1):
    print(f"\n[Test {i}]")
    print(f"Input:\n{text[:80]}...")
    
    result = parse_output(text)
    print(f"\nParsed:")
    print(f"  Answer:    {result['answer']}")
    print(f"  Reasoning: {result['reasoning'][:60]}...")
    print(f"  Valid:     {'✅' if result['valid'] else '❌'}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
valid_count = sum(1 for text in test_cases if parse_output(text)['valid'])
print(f"Valid: {valid_count}/{len(test_cases)} ({valid_count/len(test_cases)*100:.0f}%)")
print("="*70)
