# tinyllama_explainer.py
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import pandas as pd
import re
import os

class TinyLlamaExplainer:
    """TinyLlama 1.1B Chat model for explanations - Small & Effective!"""
    
    def __init__(self, model_path: str = None):
        # Set up model paths
        self.local_model_path = model_path or r"C:\Users\Mango\Desktop\nlpp\nlpp\models\tinylama"
        self.huggingface_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Check if local model exists
        self.use_local = self._check_local_model()
        
        # Set the model name to use
        self.model_name = self.local_model_path if self.use_local else self.huggingface_model
        
        self.pipe = None
        self._initialize_model()
    
    def _check_local_model(self) -> bool:
        """Check if local model exists and has required files"""
        if not os.path.exists(self.local_model_path):
            print(f"⚠️ Local model path not found: {self.local_model_path}")
            return False
        
        # Check for essential files
        essential_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        
        missing_files = []
        for file in essential_files:
            file_path = os.path.join(self.local_model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️ Local model missing files: {missing_files}")
            return False
        
        print(f"✅ Local TinyLlama model found at: {self.local_model_path}")
        return True
    
    def _initialize_model(self):
        """Initialize TinyLlama model - with local path support"""
        try:
            if self.use_local:
                print(f"🔄 Loading TinyLlama from local path: {self.local_model_path}")
            else:
                print(f"🔄 Loading TinyLlama from HuggingFace: {self.huggingface_model}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add special tokens if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate settings
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for compatibility
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float32
            )
            
            source = "local path" if self.use_local else "HuggingFace"
            print(f"✅ TinyLlama 1.1B loaded successfully from {source}!")
            
        except Exception as e:
            print(f"❌ Failed to load TinyLlama: {e}")
            print("💡 Trying fallback loading method...")
            self._try_fallback_load()
    
    def _try_fallback_load(self):
        """Try simpler loading method"""
        try:
            print("🔄 Trying fallback loading method...")
            
            # Try with simpler settings
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.float32,
                device="cpu",
                model_kwargs={"trust_remote_code": True}
            )
            
            source = "local path" if self.use_local else "HuggingFace"
            print(f"✅ TinyLlama loaded with fallback method from {source}!")
            
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            self.pipe = None
    
    def explain_results(self, sql_query: str, result_df: pd.DataFrame, execution_result: Dict[str, Any], original_question: str) -> str:
        """Generate explanation - SIMPLE VERSION"""
        
        # If model didn't load or no data, use template
        if self.pipe is None or result_df is None or result_df.empty:
            return self._simple_template_explanation(sql_query, result_df, original_question)
        
        try:
            # Prepare VERY SIMPLE context
            context = self._prepare_simple_context(result_df, sql_query)
            
            # TinyLlama chat format (it understands this well)
            prompt = f"""<|system|>
You are a helpful assistant that explains data query results in simple English.
Never write SQL code. Just explain what the numbers mean.
</|system|>

<|user|>
Question: {original_question}

Query returned these results: {context}

Explain what this means:
</|user|>

<|assistant|>
"""
            
            # Generate with conservative settings
            result = self.pipe(
                prompt,
                max_new_tokens=150,
                temperature=0.2,  # Low temperature for consistent output
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                num_return_sequences=1
            )
            
            result_list = list(result)
            full_text = result_list[0]['generated_text']
            
            # Get only the assistant's response
            if "<|assistant|>" in full_text:
                response = full_text.split("<|assistant|>")[-1].strip()
            else:
                # Try to find where prompt ends
                if prompt in full_text:
                    response = full_text[len(prompt):].strip()
                else:
                    response = full_text
            
            # Clean up
            response = self._clean_response(response)
            
            print(f"📝 TinyLlama Explanation: {response[:80]}...")
            
            # If response is bad, use template
            if len(response) < 10 or self._contains_sql(response):
                return self._simple_template_explanation(sql_query, result_df, original_question)
            
            return response
            
        except Exception as e:
            print(f"❌ TinyLlama explanation failed: {e}")
            return self._simple_template_explanation(sql_query, result_df, original_question)
    
    def _prepare_simple_context(self, result_df: pd.DataFrame, sql_query: str) -> str:
        """Prepare super simple context"""
        if result_df.empty:
            return "No data found."
        
        row_count = len(result_df)
        
        # For grouping queries (like your order priority example)
        if "GROUP BY" in sql_query.upper() and len(result_df.columns) >= 2:
            items = []
            for idx, row in result_df.iterrows()[:6]:  # Limit to 6 items
                category = str(row.iloc[0]).strip()
                value = row.iloc[1]
                
                if pd.api.types.is_numeric_dtype(result_df.iloc[:, 1]) and pd.notna(value):
                    if float(value).is_integer():
                        items.append(f"{category}: {int(value)}")
                    else:
                        items.append(f"{category}: {value:.1f}")
                else:
                    items.append(f"{category}: {value}")
            
            if items:
                return f"{row_count} groups: " + ", ".join(items)
        
        # For count queries
        elif "COUNT" in sql_query.upper() and row_count == 1:
            try:
                count_val = result_df.iloc[0, 0]
                if pd.api.types.is_numeric_dtype(result_df.iloc[:, 0]):
                    return f"Count: {int(count_val)}"
            except:
                pass
        
        # Generic
        return f"{row_count} rows of data"
    
    def _clean_response(self, text: str) -> str:
        """Clean the response text"""
        if not text:
            return ""
        
        # Remove SQL keywords
        sql_words = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT']
        for word in sql_words:
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
        
        # Remove chat format remnants
        text = re.sub(r'<\|.*?\|>', '', text)
        
        # Clean up
        text = ' '.join(text.split()).strip()
        
        if not text:
            return ""
        
        # Ensure proper punctuation
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _contains_sql(self, text: str) -> bool:
        """Check if text contains SQL"""
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY']
        text_upper = text.upper()
        return any(keyword in text_upper for keyword in sql_keywords)
    
    def _simple_template_explanation(self, sql_query: str, result_df: pd.DataFrame, question: str) -> str:
        """Simple template-based fallback"""
        if result_df is None or result_df.empty:
            return "No results found."
        
        row_count = len(result_df)
        
        # Your order priority example
        if "GROUP BY" in sql_query.upper() and len(result_df.columns) >= 2:
            items = []
            for idx, row in result_df.iterrows():
                category = str(row.iloc[0]).strip()
                value = row.iloc[1]
                
                if pd.api.types.is_numeric_dtype(result_df.iloc[:, 1]) and pd.notna(value):
                    if float(value).is_integer():
                        items.append(f"**{category}**: {int(value)}")
                    else:
                        items.append(f"**{category}**: {value:.2f}")
                else:
                    items.append(f"**{category}**: {value}")
            
            if items:
                # Add which one has most if numeric
                if pd.api.types.is_numeric_dtype(result_df.iloc[:, 1]) and row_count > 1:
                    try:
                        counts = result_df.iloc[:, 1].dropna().astype(float)
                        max_idx = counts.idxmax()
                        max_category = str(result_df.iloc[max_idx, 0]).strip()
                        max_value = counts.max()
                        
                        if max_value.is_integer():
                            return f"Results by category:\n" + "\n".join(items) + f"\n\n**{max_category}** has the most with **{int(max_value)}** items."
                        else:
                            return f"Results by category:\n" + "\n".join(items) + f"\n\n**{max_category}** has the highest value at **{max_value:.2f}**."
                    except:
                        return f"Results by category:\n" + "\n".join(items)
                else:
                    return f"Results by category:\n" + "\n".join(items)
        
        # Simple count
        if "COUNT" in sql_query.upper() and row_count == 1:
            try:
                count_val = result_df.iloc[0, 0]
                if pd.api.types.is_numeric_dtype(result_df.iloc[:, 0]):
                    return f"Found **{int(count_val)}** matching records."
            except:
                pass
        
        # Aggregation queries (SUM, AVG, MAX, MIN)
        sql_upper = sql_query.upper()
        if any(agg in sql_upper for agg in ['SUM(', 'AVG(', 'MAX(', 'MIN(']):
            if row_count == 1 and len(result_df.columns) >= 1:
                try:
                    col_name = result_df.columns[0]
                    value = result_df.iloc[0, 0]
                    
                    if pd.api.types.is_numeric_dtype(result_df.iloc[:, 0]):
                        if "SUM" in sql_upper:
                            return f"Total **{col_name}**: **{float(value):,.2f}**"
                        elif "AVG" in sql_upper:
                            return f"Average **{col_name}**: **{float(value):,.2f}**"
                        elif "MAX" in sql_upper:
                            return f"Maximum **{col_name}**: **{float(value):,.2f}**"
                        elif "MIN" in sql_upper:
                            return f"Minimum **{col_name}**: **{float(value):,.2f}**"
                except:
                    pass
        
        return f"Query returned **{row_count}** results."