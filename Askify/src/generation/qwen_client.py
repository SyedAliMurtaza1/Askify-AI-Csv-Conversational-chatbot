import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from typing import List, Dict, Any
import os
import re

logger = logging.getLogger(__name__)

class QwenClient:
    def __init__(self, model_path: str = None):
        # UPDATED PATHS - Use your v4 models
        self.lora_adapter_path = r"C:\Users\Mango\Desktop\nlpp\nlpp\models\qwen-sql-trained-v4"
        self.lora_joins_path = r"C:\Users\Mango\Desktop\nlpp\nlpp\models\qwen-sql-trained-v4\continued_training"
        self.base_model_path = r"C:\Users\Mango\Desktop\nlpp\nlpp\models\base_models\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
        
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.use_joins_adapter = False  # Track if we're using JOIN adapter
        
        # Initialize with error handling
        self._initialize_model_with_retry()
    
    def _initialize_model_with_retry(self):
        """Initialize model with multiple attempts"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"\n🚀 Attempt {attempt + 1}/{max_retries} to load model...")
                self.load_model()
                
                if self.model_loaded and self.model is not None:
                    print("✅ Model loaded successfully!")
                    return
                else:
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed with error: {e}")
                
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                else:
                    print(f"❌ All {max_retries} attempts failed!")
                    raise RuntimeError(f"Failed to load model after {max_retries} attempts: {e}")
    
    def load_model(self, use_joins_adapter: bool = False):
        """Load base model with LoRA adapter - ROBUST VERSION"""
        try:
            self.use_joins_adapter = use_joins_adapter
            
            if use_joins_adapter:
                logger.info("Loading model with JOIN adapter...")
                adapter_path = self.lora_joins_path
            else:
                logger.info("Loading model with main v4 adapter...")
                adapter_path = self.lora_adapter_path
            
            # Validate paths with detailed feedback
            self._validate_model_paths(self.base_model_path, adapter_path)
            
            print(f"📁 Loading model from: {self.base_model_path}")
            print(f"📁 Loading adapter from: {adapter_path}")
            print(f"⚙️  Device: {self.device}")
            print(f"🧠 Torch dtype: {'float16' if self.device == 'cuda' else 'float32'}")
            
            # Step 1: Load tokenizer
            print("\n1️⃣ Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, 
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"   ✅ Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
            
            # Step 2: Load base model
            print("\n2️⃣ Loading base model...")
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            device_map = "auto" if self.device == "cuda" else None
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map=device_map
            )
            
            print(f"   ✅ Base model loaded. Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
            
            # Step 3: Load LoRA adapter
            print("\n3️⃣ Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch_dtype
            )
            
            print("   ✅ LoRA adapter loaded")
            
            # Move to CPU if needed
            if self.device == "cpu":
                print("   🔄 Moving model to CPU...")
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Verify model is ready
            print("\n4️⃣ Verifying model...")
            with torch.no_grad():
                test_input = self.tokenizer("Test", return_tensors="pt")
                if self.device == "cuda":
                    test_input = test_input.to("cuda")
                output = self.model(**test_input)
                print(f"   ✅ Model verification passed. Output shape: {output.logits.shape}")
            
            self.model_loaded = True
            print(f"\n🎉 Model loaded successfully with {'JOIN' if use_joins_adapter else 'main v4'} adapter!")
            print(f"   Device: {self.device}")
            print(f"   Model: {type(self.model).__name__}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
            # Reset all attributes
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
            
            # Re-raise the exception to stop execution
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _validate_model_paths(self, base_path: str, adapter_path: str):
        """Validate that model paths exist and contain necessary files"""
        print("🔍 Validating model paths...")
        
        # Check base model
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base model path not found: {base_path}")
        
        required_base_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        base_files = os.listdir(base_path)
        print(f"   Base model files: {len(base_files)} files found")
        
        # Check adapter
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        adapter_files = os.listdir(adapter_path)
        print(f"   Adapter files: {len(adapter_files)} files found")
        
        # Check for adapter config
        if 'adapter_config.json' not in adapter_files:
            print(f"   ⚠️ Warning: adapter_config.json not found in {adapter_path}")
        
        print("   ✅ Path validation passed")
    
    def reload_with_joins(self):
        """Reload model with JOIN adapter"""
        if self.model_loaded and self.model is not None:
            print("🔄 Unloading current model...")
            del self.model
            torch.cuda.empty_cache() if self.device == "cuda" else None
        
        self.load_model(use_joins_adapter=True)
        print("🔄 Switched to JOIN adapter")
    
    def reload_with_main(self):
        """Reload model with main v4 adapter"""
        if self.model_loaded and self.model is not None:
            print("🔄 Unloading current model...")
            del self.model
            torch.cuda.empty_cache() if self.device == "cuda" else None
        
        self.load_model(use_joins_adapter=False)
        print("🔄 Switched to main v4 adapter")
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate SQL query using the model with LoRA adapter"""
        if not self.model_loaded:
            raise RuntimeError("Model is not loaded. Please initialize the model first.")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model components are not initialized.")
        
        try:
            print("\n=== QWEN GENERATION DEBUG ===")
            print(f"Using: {'JOIN adapter' if self.use_joins_adapter else 'Main v4 adapter'}")
            print(f"Prompt length: {len(prompt)} chars")
            
            # Extract numeric values from USER QUESTION only, not the whole prompt
            question_match = re.search(r"USER QUESTION\s*\n=+\s*\n*(.+?)(?:\n\n|\n=+|\nYOUR SQL|\Z)", 
                                    prompt, re.DOTALL | re.IGNORECASE)
            
            if question_match:
                user_question = question_match.group(1).strip()
                numeric_values = re.findall(r'\b\d+\b', user_question)
            else:
                numeric_values = re.findall(r'\b\d+\b', prompt[-300:])
            
            print(f"🔢 Numeric values from USER QUESTION: {numeric_values}")
            print(f"Prompt preview: {prompt[:200]}...")
            
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=False,
                add_special_tokens=True
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            print(f"Input tokens: {input_length}")
            
            # Generate with different parameters for JOIN vs non-JOIN queries
            generation_params = {
                "max_new_tokens": 350,
                "min_new_tokens": 20,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.0,
            }
            
            # Use lower temperature for JOIN queries (more deterministic)
            if self.use_joins_adapter:
                generation_params.update({
                    "temperature": 0.01,  # Very low for JOINs
                    "do_sample": False,
                    "num_beams": 1,
                })
            else:
                generation_params.update({
                    "temperature": 0.1,   # Slightly higher for general SQL
                    "do_sample": False,
                    "num_beams": 1,
                })
            
            print(f"⚙️  Generation params: {generation_params}")
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
            
            # Decode ONLY the new tokens (not the prompt)
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"Raw generated text: {generated_text}")
            
            # Extract and clean SQL
            sql_query = self._extract_and_clean_sql(generated_text)
            
            print(f"Extracted SQL (before validation): {sql_query}")
            
            # CRITICAL: Check numeric values are correct
            sql_numeric_values = re.findall(r'\b\d+\b', sql_query)
            print(f"🔢 Numeric values in SQL: {sql_numeric_values}")
            
            if numeric_values:
                for orig_value in numeric_values:
                    if orig_value not in sql_query:
                        print(f"⚠️ WARNING: Value {orig_value} from question not found in SQL!")
            
            for sql_value in sql_numeric_values:
                if sql_value not in numeric_values:
                    print(f"ℹ️ INFO: SQL contains {sql_value} not mentioned in question")
            
            # Validate SQL structure
            sql_query = self._fix_sql_structure(sql_query)
            
            print(f"Final SQL: {sql_query}")
            
            # Check if JOIN was used (if using JOIN adapter)
            if self.use_joins_adapter and 'JOIN' not in sql_query.upper():
                print("⚠️ JOIN adapter used but no JOIN generated!")
            
            print("=" * 50 + "\n")
            
            return sql_query
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _fix_sql_structure(self, sql_query: str) -> str:
        """Fix SQL structure to ensure FROM comes before WHERE"""
        sql_upper = sql_query.upper()
        
        # Check if both FROM and WHERE exist
        if 'FROM' in sql_upper and 'WHERE' in sql_upper:
            where_pos = sql_upper.find('WHERE')
            from_pos = sql_upper.find('FROM')
            
            # If WHERE comes before FROM, fix it
            if where_pos < from_pos:
                print("⚠️ Fixing SQL structure: WHERE before FROM")
                
                # Extract components
                select_match = re.search(r'(SELECT\s+.*?)\s+WHERE', sql_query, re.IGNORECASE)
                where_match = re.search(r'WHERE\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
                from_match = re.search(r'FROM\s+(\S+)', sql_query, re.IGNORECASE)
                
                if select_match and where_match and from_match:
                    select_part = select_match.group(1).strip()
                    where_part = where_match.group(1).strip()
                    from_part = from_match.group(1).strip()
                    
                    # Rebuild correctly
                    sql_query = f"{select_part} FROM {from_part} WHERE {where_part}"
                    print(f"✓ Fixed SQL: {sql_query}")
        
        # If WHERE exists but no FROM, add FROM
        elif 'WHERE' in sql_upper and 'FROM' not in sql_upper:
            print("⚠️ Adding missing FROM clause")
            sql_query = re.sub(
                r'\s+WHERE\s+',
                ' FROM data WHERE ',
                sql_query,
                flags=re.IGNORECASE
            )
            print(f"✓ Fixed SQL: {sql_query}")
        
        return sql_query

    def _extract_and_clean_sql(self, generated_text: str) -> str:
        """Extract and clean SQL query from generated text"""
        # Remove markdown code blocks
        generated_text = re.sub(r'```sql\s*', '', generated_text)
        generated_text = re.sub(r'```\s*', '', generated_text)
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^SQL:\s*',
            r'^Query:\s*',
            r'^Answer:\s*',
            r'^Response:\s*',
            r"^Here's the SQL:\s*",
            r'^The SQL query is:\s*',
        ]
        
        for prefix in prefixes_to_remove:
            generated_text = re.sub(prefix, '', generated_text, flags=re.IGNORECASE)
        
        # Find SELECT statement
        select_match = re.search(
            r'(SELECT\s+.+?)(?:\s*;|\s*Note:|\s*Explanation:|\s*\n\n|\s*$)', 
            generated_text, 
            re.IGNORECASE | re.DOTALL
        )
        
        if select_match:
            sql_query = select_match.group(1).strip()
        else:
            # Try to find SQL anywhere
            for terminator in [';', '\n\n', 'Note:', 'Explanation:', 'This query']:
                if terminator in generated_text:
                    sql_query = generated_text.split(terminator)[0].strip()
                    break
            else:
                sql_query = generated_text.strip()
        
        # Clean up
        sql_query = self._clean_sql_preserve_numbers(sql_query)
        
        return sql_query

    def _clean_sql_preserve_numbers(self, sql: str) -> str:
        """Clean SQL while preserving ALL numeric values"""
        # Remove trailing semicolons
        sql = sql.rstrip(';').strip()
        
        # Remove comments carefully
        sql = re.sub(r'--[^\n]*', '', sql)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def generate_sql_for_question(self, question: str, columns: str, use_joins: bool = False) -> str:
        """Helper method to generate SQL for a natural language question"""
        if use_joins != self.use_joins_adapter:
            if use_joins:
                self.reload_with_joins()
            else:
                self.reload_with_main()
        
        prompt = f"Question: {question}\nColumns: {columns}\nSQL:"
        return self.generate_response(prompt)


# Test function
def test_qwen_model():
    """Test function to verify Qwen model is working correctly"""
    print("\n" + "=" * 60)
    print("TESTING QWEN v4 MODEL")
    print("=" * 60)
    
    try:
        # Initialize client with main adapter
        print("🚀 Initializing QwenClient...")
        client = QwenClient()
        
        if not client.model_loaded:
            print("❌ Model failed to load")
            return False
        
        print("✅ Model loaded successfully!\n")
        
        # Test basic SQL generation
        print("🧪 Testing basic SQL generation...")
        test_prompt = "Question: show employees with salary > 5000\nColumns: employees: id, name, salary, department\nSQL:"
        
        try:
            sql = client.generate_response(test_prompt)
            print(f"🤖 Generated SQL: {sql}")
            
            if 'SELECT' in sql.upper():
                print("✅ SQL generation successful!")
                return True
            else:
                print("⚠️ Generated output doesn't look like SQL")
                return False
                
        except Exception as e:
            print(f"❌ SQL generation test failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = test_qwen_model()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED")
        print("=" * 60)