"""
Quick verification script to check if models are trained and ready
Run this before the TA demo!
"""

import os
from pathlib import Path

def check_models(task='subject'):
    """Check if all models are trained and saved"""
    
    print("=" * 70)
    print("🔍 MODEL VERIFICATION CHECK")
    print("=" * 70)
    
    model_dir = f'./saved_models/{task}'
    
    # Required files
    required_files = {
        'random_forest.pkl': 'Random Forest Model',
        'svm.pkl': 'SVM Model',
        'lstm_model.pt': 'LSTM Model',
        'lstm_config.pkl': 'LSTM Config',
        'cnn_model.pt': 'CNN Model',
        'cnn_config.pkl': 'CNN Config',
        'tfidf_vectorizer.pkl': 'TF-IDF Vectorizer',
        'vocab.pkl': 'Vocabulary',
        'label_encoder.pkl': 'Label Encoder'
    }
    
    # Check directory
    print(f"\n📁 Checking directory: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"❌ Directory not found!")
        print(f"\n⚠️  MODELS NOT TRAINED YET!")
        print(f"\nTo train models:")
        print(f"1. Run: Train_Models_Save_For_Prediction.ipynb")
        print(f"2. Or upload to Google Colab and run there")
        return False
    
    print(f"✅ Directory exists")
    
    # Check files
    print(f"\n📋 Checking required files:")
    all_found = True
    total_size = 0
    
    for filename, description in required_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            total_size += size_mb
            print(f"  ✅ {description:25} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {description:25} (MISSING!)")
            all_found = False
    
    print(f"\n💾 Total size: {total_size:.2f} MB")
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_found:
        print("✅ ALL MODELS READY!")
        print("🎉 You can run the prediction app now!")
        print("\nRun: streamlit run prediction_app.py")
    else:
        print("❌ SOME MODELS MISSING!")
        print("⚠️  Please train models first!")
        print("\nRun: Train_Models_Save_For_Prediction.ipynb")
    print("=" * 70)
    
    return all_found

def check_all_tasks():
    """Check all three classification tasks"""
    print("\n🔍 CHECKING ALL CLASSIFICATION TASKS\n")
    
    tasks = ['subject', 'difficulty', 'question_type']
    results = {}
    
    for task in tasks:
        print(f"\n{'─' * 70}")
        print(f"Task: {task.upper()}")
        print(f"{'─' * 70}")
        results[task] = check_models(task)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    for task in tasks:
        status = "✅ Ready" if results[task] else "❌ Not trained"
        print(f"{task:20} : {status}")
    
    print("=" * 70)
    
    ready_count = sum(results.values())
    print(f"\n{ready_count}/{len(tasks)} tasks are ready for prediction")
    
    if ready_count == 0:
        print("\n⚠️  No models trained yet!")
        print("Please run: Train_Models_Save_For_Prediction.ipynb")
    elif ready_count < len(tasks):
        print("\n💡 Tip: Train remaining tasks using the notebook")
        print("   Just change TARGET_COLUMN = 'difficulty' or 'question_type'")
    else:
        print("\n🎉 ALL TASKS READY! You're fully prepared for the demo!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        task = sys.argv[1]
        check_models(task)
    else:
        # Check subject classification by default
        check_models('subject')
        
        # Ask if user wants to check all
        print("\n" + "=" * 70)
        response = input("Check other tasks too? (y/n): ").strip().lower()
        if response == 'y':
            check_all_tasks()
