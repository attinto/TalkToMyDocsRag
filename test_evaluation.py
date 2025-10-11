"""
Simple test script to verify the evaluation system is working correctly.
Run this before doing a full evaluation to catch any import or setup issues.
"""

import os
import sys


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import ragas
        print("✅ ragas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ragas: {e}")
        return False
    
    try:
        import pandas
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pandas: {e}")
        return False
    
    try:
        import datasets
        print("✅ datasets imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import datasets: {e}")
        return False
    
    try:
        from src.evaluation.rag_evaluator import RAGEvaluator
        print("✅ RAGEvaluator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RAGEvaluator: {e}")
        return False
    
    return True


def test_environment():
    """Test that required environment variables are set."""
    print("\nTesting environment variables...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print("✅ OPENAI_API_KEY is set")
    return True


def test_eval_dataset():
    """Test that the evaluation dataset exists and is valid."""
    print("\nTesting evaluation dataset...")
    
    dataset_path = "src/data/eval_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Evaluation dataset not found at {dataset_path}")
        return False
    
    print(f"✅ Evaluation dataset found at {dataset_path}")
    
    try:
        import json
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Dataset is not a list")
            return False
        
        print(f"✅ Dataset contains {len(data)} questions")
        
        # Check structure of first item
        if len(data) > 0:
            required_keys = ["question", "ground_truth_answer", "ground_truth_context"]
            first_item = data[0]
            
            for key in required_keys:
                if key not in first_item:
                    print(f"❌ Missing required key '{key}' in dataset items")
                    return False
            
            print("✅ Dataset structure is valid")
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False
    
    return True


def test_basic_rag():
    """Test that we can import and run a basic RAG."""
    print("\nTesting Basic RAG module...")
    
    try:
        from src.chatbot.rags.basic_rag import execute_rag, execute_rag_with_context
        print("✅ Basic RAG functions imported successfully")
        
        # Test that functions are callable
        if not callable(execute_rag):
            print("❌ execute_rag is not callable")
            return False
        
        if not callable(execute_rag_with_context):
            print("❌ execute_rag_with_context is not callable")
            return False
        
        print("✅ Basic RAG functions are callable")
        
    except ImportError as e:
        print(f"❌ Failed to import Basic RAG: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("RAG Evaluation System - Pre-flight Check")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_environment()
    all_passed &= test_eval_dataset()
    all_passed &= test_basic_rag()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! You're ready to run evaluation.")
        print("\nTry running:")
        print("  python -m src.main evaluate -s basic")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nIf you're missing dependencies, install them with:")
        print("  uv pip install ragas datasets pandas")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
