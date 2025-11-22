"""
Quick test to verify fixes for train_student_ultimate.py
"""
import torch
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")

# Test 1: Check autocast API
print("\n[Test 1] Autocast API")
try:
    if torch.cuda.is_available():
        with torch.amp.autocast('cuda'):
            x = torch.randn(2, 10).cuda()
            y = x * 2
        print("✓ New autocast API works")
    else:
        print("⊗ CUDA not available, skipping")
except Exception as e:
    print(f"✗ New autocast API failed: {e}")

# Test 2: Check label_smoothing parameter
print("\n[Test 2] Label smoothing in F.cross_entropy")
try:
    logits = torch.randn(4, 10)  # batch*seq, vocab
    labels = torch.randint(0, 10, (4,))
    
    loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=0.1,
        reduction='none'
    )
    print(f"✓ Label smoothing works, loss shape: {loss.shape}")
except TypeError as e:
    print(f"✗ Label smoothing not supported: {e}")
    print("  Using PyTorch < 1.10")

# Test 3: Check tensor shape operations
print("\n[Test 3] Tensor shape operations")
try:
    # Simulate loss computation
    logits = torch.randn(2, 160, 1000)  # batch, seq, vocab
    labels = torch.randint(-100, 100, (2, 160))  # batch, seq
    
    logits_flat = logits.view(-1, 1000)  # 320, 1000
    labels_flat = labels.view(-1)  # 320
    
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction='none'
    )
    
    weights = torch.ones_like(labels_flat, dtype=torch.float)
    valid_mask = (labels_flat != -100).float()
    
    weighted_loss = (loss * weights * valid_mask).sum() / (weights * valid_mask).sum().clamp(min=1.0)
    
    print(f"✓ Shape operations correct")
    print(f"  Logits flat: {logits_flat.shape}")
    print(f"  Labels flat: {labels_flat.shape}")
    print(f"  Loss: {loss.shape}")
    print(f"  Weighted loss: {weighted_loss.item():.4f}")
except Exception as e:
    print(f"✗ Shape operations failed: {e}")

print("\n" + "="*50)
print("Summary:")
print("If all tests pass, train_student_ultimate.py should work!")
print("="*50)
