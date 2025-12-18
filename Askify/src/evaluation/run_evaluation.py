import json
import pandas as pd
import os
from evaluation_rag_system import EvaluationRAGSystem
from evaluator import RAGEvaluator

def main():
    # Load your QA dataset
    with open(r'C:\Users\Mango\Desktop\nlpp\nlpp\data\csv_qa_dataset.json', 'r') as f:
        qa_dataset = json.load(f)
    
    # Your 4 CSV files
    csv_files = [
        # r"C:\Users\Mango\Desktop\nlpp\nlpp\data\CustomerCSV.csv",
        r"C:\Users\Mango\Desktop\nlpp\nlpp\data\SalesCSV.csv", 
        r"C:\Users\Mango\Desktop\nlpp\nlpp\data\ProductCSV.csv",
        r"C:\Users\Mango\Desktop\nlpp\nlpp\data\Titanic-Dataset.csv"
    ]
    
    # Create results directory
    results_dir = r'C:\Users\Mango\Desktop\nlpp\nlpp\data\evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    all_comparisons = []
    
    # Test each CSV file
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV file not found: {csv_path}, skipping...")
            continue
            
        print(f"\n🔍 Testing with: {csv_path}")
        
        # Initialize systems for this CSV
        rag_system = EvaluationRAGSystem(use_rag=True)
        baseline_system = EvaluationRAGSystem(use_rag=False)
        
        # Load CSV data
        try:
            rag_system.initialize_system(csv_path)
            baseline_system.initialize_system(csv_path)
        except Exception as e:
            print(f"❌ Failed to initialize system for {csv_path}: {e}")
            continue
        
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Run evaluation
        rag_results = []
        baseline_results = []
        
        print("🚀 Starting RAG vs Baseline Evaluation...")
        
        # Filter relevant questions for this CSV
        relevant_questions = [qa for qa in qa_dataset if _is_question_relevant(qa, csv_path)]
        total_questions = len(relevant_questions)
        
        if total_questions == 0:
            print(f"⚠️  No relevant questions found for {os.path.basename(csv_path)}")
            continue
            
        print(f"📋 Found {total_questions} relevant questions for this CSV")
        
        for i, qa_pair in enumerate(relevant_questions):
            question = qa_pair['question']
            
            print(f"   [{i+1}/{total_questions}] Processing: {question[:60]}...")
            
            # Get RAG response
            try:
                rag_response = rag_system.answer_question(question)
                rag_eval = evaluator.evaluate_response(question, qa_pair, rag_response, csv_path)
                rag_results.append(rag_eval)
            except Exception as e:
                print(f"   ❌ RAG failed for question {i+1}: {e}")
                rag_results.append(_create_failed_evaluation(question, csv_path, 'RAG'))
            
            # Get Baseline response
            try:
                baseline_response = baseline_system.answer_question(question)
                baseline_eval = evaluator.evaluate_response(question, qa_pair, baseline_response, csv_path)
                baseline_results.append(baseline_eval)
            except Exception as e:
                print(f"   ❌ Baseline failed for question {i+1}: {e}")
                baseline_results.append(_create_failed_evaluation(question, csv_path, 'Baseline'))
        
        if rag_results:  # Only compare if we have results
            # Compare results for this CSV
            comparison_df = evaluator.run_comparison(rag_results, baseline_results)
            comparison_df['csv_file'] = csv_path
            all_comparisons.append(comparison_df)
            
            # Save individual results
            _save_individual_results(rag_results, baseline_results, csv_path)
    
    # Combine all comparisons
    if all_comparisons:
        final_comparison = pd.concat(all_comparisons, ignore_index=True)
        final_comparison.to_csv(os.path.join(results_dir, 'rag_vs_baseline_comparison.csv'), index=False)
        
        # Print summary
        print("\n📊 FINAL EVALUATION SUMMARY:")
        for csv in final_comparison['csv_file'].unique():
            csv_data = final_comparison[final_comparison['csv_file'] == csv]
            csv_name = os.path.basename(csv)
            
            print(f"\n📁 {csv_name}:")
            print(f"   Questions Tested: {len(csv_data)}")
            print(f"   RAG Average Score: {csv_data['rag_overall_score'].mean():.3f}")
            print(f"   Baseline Average Score: {csv_data['baseline_overall_score'].mean():.3f}")
            print(f"   Average Improvement: {csv_data['improvement'].mean():.3f}")
            
            # Calculate win rates
            rag_wins = len(csv_data[csv_data['rag_overall_score'] > csv_data['baseline_overall_score']])
            baseline_wins = len(csv_data[csv_data['rag_overall_score'] < csv_data['baseline_overall_score']])
            ties = len(csv_data[csv_data['rag_overall_score'] == csv_data['baseline_overall_score']])
            
            print(f"   RAG Wins: {rag_wins} ({rag_wins/len(csv_data)*100:.1f}%)")
            print(f"   Baseline Wins: {baseline_wins} ({baseline_wins/len(csv_data)*100:.1f}%)")
            print(f"   Ties: {ties} ({ties/len(csv_data)*100:.1f}%)")
    
    print("✅ Evaluation completed!")

def _is_question_relevant(qa_pair, csv_path):
    """Check if question is relevant to the current CSV"""
    csv_name = os.path.basename(csv_path)
    csv_name_no_ext = csv_name.replace('.csv', '')
    
    for source in qa_pair.get('sources', []):
        source_name = source.get('csv_name', '')
        # Check both with and without extension, and partial matches
        if (source_name == csv_name or 
            source_name == csv_name_no_ext or
            csv_name_no_ext in source_name):
            return True
    return False

def _create_failed_evaluation(question: str, csv_path: str, mode: str) -> dict:
    """Create evaluation dict for failed responses"""
    return {
        'question': question,
        'csv_file': csv_path,
        'factuality_score': 0.1,
        'completeness_score': 0.1,
        'faithfulness_score': 0.1 if mode == 'Baseline' else 0.2,
        'safety_score': 1.0,  # Failed but safe
        'sql_valid': False,
        'execution_success': False,
        'retrieved_sources_count': 0,
        'mode': mode,
        'overall_score': 0.1
    }

def _save_individual_results(rag_results, baseline_results, csv_path):
    """Save individual response files in separate folders with absolute paths"""
    csv_name = os.path.basename(csv_path).replace('.csv', '')
    
    # Create main results directory
    results_dir = r'C:\Users\Mango\Desktop\nlpp\nlpp\data\evaluation_results'
    
    # Create subdirectories
    rag_dir = os.path.join(results_dir, 'rag_responses')
    baseline_dir = os.path.join(results_dir, 'baseline_responses')
    os.makedirs(rag_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Save RAG responses in rag_responses folder
    rag_filename = os.path.join(rag_dir, f'rag_{csv_name}_responses.json')
    with open(rag_filename, 'w') as f:
        json.dump(rag_results, f, indent=2)
    print(f"   ✅ Saved RAG responses: {rag_filename}")
    
    # Save Baseline responses in baseline_responses folder  
    baseline_filename = os.path.join(baseline_dir, f'baseline_{csv_name}_responses.json')
    with open(baseline_filename, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print(f"   ✅ Saved Baseline responses: {baseline_filename}")

if __name__ == "__main__":
    main()