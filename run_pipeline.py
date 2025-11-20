"""
Master Script - Run Complete Training & Evaluation Pipeline
Author: Enhanced for thesis
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(description, command, log_file=None):
    """Run a command and log output"""
    print_header(description)
    print(f"Command: {command}\n")
    
    start_time = time.time()
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
        else:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True
            )
        
        elapsed = time.time() - start_time
        print(f"✅ {description} completed in {elapsed/60:.1f} minutes")
        
        if log_file:
            print(f"   Log saved to: {log_file}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed/60:.1f} minutes")
        print(f"   Error: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if required file exists"""
    if os.path.exists(filepath):
        print(f"✅ Found: {description}")
        return True
    else:
        print(f"❌ Missing: {description}")
        print(f"   Path: {filepath}")
        return False

def main():
    """Main pipeline execution"""
    print_header("VQA THESIS - COMPLETE TRAINING & EVALUATION PIPELINE")
    
    print("📋 Pipeline Steps:")
    print("  1. Analyze current model issues")
    print("  2. Train with curriculum learning")
    print("  3. Evaluate on test set")
    print("  4. Visualize results")
    print("  5. Generate thesis report\n")
    
    # Configuration
    SAVE_DIR = "/kaggle/working"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check prerequisites
    print_header("STEP 0: Checking Prerequisites")
    
    required_files = {
        "model.py": "VQA Model Architecture",
        "train_student_ultimate.py": "Training Script",
        "eval_adaptive_v3.py": "Evaluation Script",
        "analyze_issues.py": "Analysis Script"
    }
    
    all_exist = True
    for file, desc in required_files.items():
        if not check_file_exists(file, desc):
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some required files are missing. Please ensure all scripts are in the current directory.")
        return
    
    print("\n✅ All required files found!\n")
    
    # Ask user what to run
    print("What would you like to run?")
    print("  1. Full pipeline (analyze + train + eval + visualize)")
    print("  2. Only analysis")
    print("  3. Only training")
    print("  4. Only evaluation")
    print("  5. Training + Evaluation")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    success = True
    
    # Step 1: Analysis
    if choice in ['1', '2']:
        if run_command(
            "STEP 1: Analyzing Current Model Issues",
            "python analyze_issues.py",
            f"{SAVE_DIR}/analysis_log_{timestamp}.txt"
        ):
            print("\n📊 Review the analysis output above to understand current problems.\n")
            if choice == '2':
                return
        else:
            print("\n⚠️  Analysis failed, but continuing...\n")
    
    # Step 2: Training
    if choice in ['1', '3', '5']:
        print("\n" + "⚠️ "*35)
        print("⚠️  IMPORTANT: Training will take 6-8 hours on GPU!")
        print("⚠️  Make sure you have:")
        print("⚠️    - Sufficient GPU memory (at least 16GB)")
        print("⚠️    - Kaggle session won't timeout")
        print("⚠️    - Teacher data is accessible")
        print("⚠️ "*35 + "\n")
        
        if choice != '1':
            confirm = input("Continue with training? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Training cancelled.")
                if choice == '3':
                    return
                success = False
        
        if success or choice == '1':
            if not run_command(
                "STEP 2: Training with Curriculum Learning",
                "python train_student_ultimate.py",
                f"{SAVE_DIR}/training_log_{timestamp}.txt"
            ):
                success = False
                print("\n❌ Training failed. Check the log file for details.")
                if choice == '3':
                    return
    
    # Step 3: Evaluation
    if choice in ['1', '4', '5'] and success:
        if not run_command(
            "STEP 3: Evaluating on Test Set",
            "python eval_adaptive_v3.py",
            f"{SAVE_DIR}/eval_log_{timestamp}.txt"
        ):
            success = False
            print("\n❌ Evaluation failed. Check the log file for details.")
            if choice == '4':
                return
    
    # Step 4: Visualization
    if choice == '1' and success:
        print_header("STEP 4: Visualizing Results")
        
        # Check if results exist
        train_log = f"{SAVE_DIR}/train_val_log_ultimate.csv"
        eval_results = f"{SAVE_DIR}/eval_adaptive_v3_results.csv"
        
        if os.path.exists(train_log):
            run_command(
                "Generating Training Visualization",
                f"python visualize_results.py train {train_log}",
                None
            )
        else:
            print(f"⚠️  Training log not found: {train_log}")
        
        if os.path.exists(eval_results):
            run_command(
                "Generating Evaluation Visualization",
                f"python visualize_results.py eval {eval_results}",
                None
            )
        else:
            print(f"⚠️  Evaluation results not found: {eval_results}")
    
    # Final summary
    print_header("PIPELINE COMPLETED")
    
    if success:
        print("✅ All steps completed successfully!\n")
        
        print("📁 Output Files:")
        output_files = [
            (f"{SAVE_DIR}/vqa_student_best_ultimate.pt", "Best model checkpoint"),
            (f"{SAVE_DIR}/train_val_log_ultimate.csv", "Training logs"),
            (f"{SAVE_DIR}/eval_adaptive_v3_results.csv", "Evaluation results"),
            (f"{SAVE_DIR}/training_progress.png", "Training visualization"),
            (f"{SAVE_DIR}/evaluation_results.png", "Evaluation visualization")
        ]
        
        for filepath, description in output_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024*1024)
                print(f"  ✅ {description}: {filepath} ({size:.1f} MB)")
            else:
                print(f"  ⚠️  {description}: Not found")
        
        print("\n📊 Next Steps:")
        print("  1. Review evaluation metrics in CSV file")
        print("  2. Check training/evaluation visualizations")
        print("  3. Compare with baseline model")
        print("  4. Generate qualitative examples for thesis")
        print("  5. Prepare defense presentation")
        
        print("\n📖 Documentation:")
        print("  • THESIS_GUIDE.md       - Complete guide")
        print("  • SOLUTION_SUMMARY.md   - Solution overview")
        print("  • QUICK_REFERENCE.txt   - Quick commands")
        
    else:
        print("❌ Pipeline completed with errors.\n")
        print("🔧 Troubleshooting:")
        print("  1. Check log files in:", SAVE_DIR)
        print("  2. Verify data paths are correct")
        print("  3. Ensure GPU memory is sufficient")
        print("  4. Review error messages above")
        print("  5. Consult THESIS_GUIDE.md for help")
    
    print("\n" + "="*70)
    print("  🎓 Good luck with your thesis! 🚀")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        print("   Partial results may be available in /kaggle/working/")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("   Please report this error and check the documentation.")
        sys.exit(1)
