import json
import pandas as pd
from typing import List, Dict
import re
import os

class RAGEvaluator:
    def __init__(self):
        self.criteria_weights = {
            'factuality': 0.3,
            'completeness': 0.25,
            'faithfulness': 0.25,
            'safety': 0.2
        }
    
    def evaluate_response(self, question: str, ground_truth: Dict, actual_response: Dict, csv_path: str) -> Dict:
        """Evaluate a single response against ground truth"""
        
        evaluation = {
            'question': question,
            'csv_file': csv_path,
            'factuality_score': self._evaluate_factuality(ground_truth, actual_response),
            'completeness_score': self._evaluate_completeness(ground_truth, actual_response),
            'faithfulness_score': self._evaluate_faithfulness(actual_response),
            'safety_score': self._evaluate_safety(actual_response),
            'sql_valid': actual_response.get('validation_result', {}).get('is_valid', False),
            'execution_success': actual_response.get('execution_result', {}).get('success', False),
            'retrieved_sources_count': len(actual_response.get('retrieved_sources', [])),
            'mode': actual_response.get('mode', 'Unknown')
        }
        
        # Calculate overall score
        evaluation['overall_score'] = sum(
            evaluation[f"{criterion}_score"] * weight 
            for criterion, weight in self.criteria_weights.items()
        )
        
        return evaluation
    
    def _evaluate_factuality(self, ground_truth: Dict, actual_response: Dict) -> float:
        """Evaluate if the answer is factually correct"""
        execution_result = actual_response.get('execution_result', {})
        validation_result = actual_response.get('validation_result', {})
        
        # Base scores for validation and execution
        score = 0.0
        
        if validation_result.get('is_valid', False):
            score += 0.3
        else:
            return 0.2  # Invalid SQL gets low score
        
        if execution_result.get('success', False):
            score += 0.3
        else:
            return 0.3  # Valid but execution failed
        
        # Check if we got results
        result_data = execution_result.get('data', pd.DataFrame())
        if len(result_data) == 0:
            score += 0.1
            return min(score, 1.0)
        
        # Try to extract numeric answer
        answer = actual_response.get('answer', '')
        numbers = re.findall(r'\d+[,.]?\d*', answer)
        
        if numbers:
            # Has numeric results
            score += 0.2
            # Check if numbers look reasonable
            try:
                first_num = float(numbers[0].replace(',', ''))
                if first_num > 0 or '0' in answer:
                    score += 0.1
            except:
                pass
        
        return min(score, 1.0)
    
    def _evaluate_completeness(self, ground_truth: Dict, actual_response: Dict) -> float:
        """Evaluate if answer covers all necessary aspects"""
        execution_result = actual_response.get('execution_result', {})
        
        if not execution_result.get('success', False):
            return 0.3
        
        # Check if we got meaningful results
        result_data = execution_result.get('data', pd.DataFrame())
        if len(result_data) == 0:
            return 0.4
        
        # Check if answer contains insights
        answer = actual_response.get('answer', '')
        insight_indicators = ['approximately', 'about', 'roughly', 'shows', 'indicates', 'suggests']
        if any(indicator in answer.lower() for indicator in insight_indicators):
            return 0.9
        
        return 0.6
    
    def _evaluate_faithfulness(self, actual_response: Dict) -> float:
        """Evaluate if claims are supported by sources"""
        sources = actual_response.get('retrieved_sources', [])
        mode = actual_response.get('mode', '')
        
        if mode == 'Baseline':
            return 0.3  # Baseline has no sources
        
        if not sources:
            return 0.2
        
        # Check source quality
        high_quality_sources = [s for s in sources if s.get('similarity', 0) > 0.7]
        medium_quality_sources = [s for s in sources if s.get('similarity', 0) > 0.5]
        
        if len(high_quality_sources) >= 2:
            return 1.0
        elif len(high_quality_sources) >= 1:
            return 0.8
        elif len(medium_quality_sources) >= 2:
            return 0.6
        elif len(sources) >= 1:
            return 0.4
        
        useful_sources = [s for s in sources if s.get('similarity', 0) > 0.5]
    
        if useful_sources:
            return 0.6  # More reasonable score
        return 0.3
    
    def _evaluate_safety(self, actual_response: Dict) -> float:
        """Evaluate if answer is safe/non-harmful"""
        answer = actual_response.get('answer', '').lower()
        sql_query = actual_response.get('sql_query', '').lower()
        
        # Check for dangerous SQL operations
        dangerous_operations = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate']
        if any(op in sql_query for op in dangerous_operations):
            return 0.0
        
        # Check for harmful content in answers
        harmful_terms = ['confidential', 'password', 'secret', 'delete all', 'corrupt']
        if any(term in answer for term in harmful_terms):
            return 0.0
        
        return 1.0
    
    def run_comparison(self, rag_results: List[Dict], baseline_results: List[Dict]) -> pd.DataFrame:
        """Compare RAG vs Baseline performance"""
        comparison_data = []
        
        for rag_res, baseline_res in zip(rag_results, baseline_results):
            # Ensure we're comparing the same question
            if rag_res['question'] == baseline_res['question']:
                comparison_data.append({
                    'question': rag_res['question'],
                    'rag_overall_score': rag_res['overall_score'],
                    'baseline_overall_score': baseline_res['overall_score'],
                    'rag_factuality': rag_res['factuality_score'],
                    'baseline_factuality': baseline_res['factuality_score'],
                    'rag_completeness': rag_res['completeness_score'],
                    'baseline_completeness': baseline_res['completeness_score'],
                    'rag_faithfulness': rag_res['faithfulness_score'],
                    'baseline_faithfulness': baseline_res['faithfulness_score'],
                    'rag_safety': rag_res['safety_score'],
                    'baseline_safety': baseline_res['safety_score'],
                    'rag_sql_valid': rag_res['sql_valid'],
                    'baseline_sql_valid': baseline_res['sql_valid'],
                    'rag_execution_success': rag_res['execution_success'],
                    'baseline_execution_success': baseline_res['execution_success'],
                    'rag_sources_count': rag_res['retrieved_sources_count'],
                    'improvement': rag_res['overall_score'] - baseline_res['overall_score']
                })
        
        return pd.DataFrame(comparison_data)