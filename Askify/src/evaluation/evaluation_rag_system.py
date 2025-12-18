import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.data_processor import DataProcessor
from src.rag.csv_chunker import CSVChunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_engineer import PromptEngineer
from src.generation.qwen_client import QwenClient
from src.utils.sql_executor import SQLExecutor
from src.utils.sql_validator import SQLValidator
from src.utils.lama_explainer import TinyLlamaExplainer

class EvaluationRAGSystem:
    def __init__(self, use_rag=True):
        self.use_rag = use_rag
        self.processor = DataProcessor()
        self.chunker = CSVChunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.context_builder = ContextBuilder()
        self.prompt_engineer = PromptEngineer()
        self.sql_executor = SQLExecutor()
        self.sql_validator = SQLValidator()
        
        # Initialize components
        self.qwen_client = None
        self.nl_explainer = TinyLlamaExplainer()
        self.model_loaded = False
        self.current_csv = None
        
        # For evaluation tracking
        self.retrieved_sources = []
    
    def initialize_system(self, csv_path: str):
        """Initialize with CSV data"""
        self.current_csv = csv_path
        df = self.processor.load_csv(csv_path)
        
        if self.use_rag:
            # Build RAG index
            print(f"   Building RAG index for {csv_path}...")
            chunks = self.chunker.chunk_dataframe(df)
            self.vector_store.add_chunks(chunks, self.embedder)
            print(f"   ✅ RAG index built with {len(chunks)} chunks")
        
        self.df = df
        self.initialize_model()
        return df
    
    def initialize_model(self):
        """Initialize the Qwen model"""
        try:
            if self.qwen_client is None:
                self.qwen_client = QwenClient()
                self.model_loaded = self.qwen_client.model_loaded
                if self.model_loaded:
                    print("   ✅ Qwen model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load Qwen model: {e}")
            self.model_loaded = False
            raise e
    

    def answer_question(self, question: str):
        """Answer question with or without RAG"""
        if not self.model_loaded:
            raise RuntimeError("AI model is not loaded yet.")
        
        if self.use_rag:
            # RAG MODE: Use enhanced retrieval with multiple search terms
            print(f"   🔍 Enhanced search for: {question[:50]}...")
            
            # Get enhanced search terms from context builder
            search_terms = self.context_builder.get_enhanced_search_terms(question)
            print(f"   📝 Search terms: {search_terms[:3]}")
            
            # Search with multiple terms and combine results
            all_results = []
            for term in search_terms[:3]:  # Use top 3 search terms
                try:
                    results = self.vector_store.search(term, self.embedder, top_k=3)
                    all_results.extend(results)
                    print(f"   → '{term}': found {len(results)} chunks")
                except Exception as e:
                    print(f"   ⚠️  Search error for '{term}': {e}")
            
            # Deduplicate chunks
            retrieved_chunks = self._deduplicate_chunks(all_results)
            print(f"   ✅ Total unique chunks: {len(retrieved_chunks)}")
            
            # Build context with enhanced analysis
            context = self.context_builder.build_context(retrieved_chunks, question)
            
            # Store sources for evaluation
            self.retrieved_sources = [
                {
                    'content': chunk['content'][:200] + "...",
                    'metadata': chunk['metadata'],
                    'similarity': chunk.get('similarity', 0),
                    'search_term': chunk.get('search_term', 'main')
                }
                for chunk in retrieved_chunks[:5]  # Store top 5
            ]
        else:
            # BASELINE MODE: No RAG (unchanged)
            context = "No additional context provided."
            self.retrieved_sources = []
        
        # Generate SQL
        prompt = self.prompt_engineer.create_prompt(context, question)
        sql_query = self.qwen_client.generate_response(prompt)
        
        # Validate SQL
        validation_result = self.sql_validator.validate_sql(sql_query, list(self.df.columns))
        
        # Execute SQL
        execution_result = self.sql_executor.execute_sql(sql_query, self.df)
        
        # Generate natural language explanation
        if execution_result['success']:
            explanation = self.nl_explainer.explain_results(
                sql_query, execution_result['data'], execution_result, question
            )
        else:
            explanation = f"Query execution failed: {execution_result.get('error', 'Unknown error')}"
        
        return {
            'question': question,
            'sql_query': sql_query,
            'answer': explanation,
            'execution_result': execution_result,
            'validation_result': validation_result,
            'retrieved_sources': self.retrieved_sources,
            'csv_file': self.current_csv,
            'mode': 'RAG' if self.use_rag else 'Baseline'
        }

    def _deduplicate_chunks(self, chunks):
        """Remove duplicate chunks based on content"""
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create a signature from content and metadata
            content_preview = chunk['content'][:100]  # First 100 chars
            metadata_str = str(chunk.get('metadata', {}))
            signature = f"{content_preview}_{metadata_str}"
            
            if signature not in seen:
                seen.add(signature)
                unique_chunks.append(chunk)
        
        # Sort by similarity
        unique_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return unique_chunks