import sqlite3
import pandas as pd
import re
from typing import Dict, Any, List, Tuple
import logging
from difflib import get_close_matches

logger = logging.getLogger(__name__)

class SQLExecutor:
    def __init__(self):
        self.table_name = "data"  # Default table name
    
    def execute_sql(self, sql_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute SQL query with robust error handling and column mapping"""
        try:
            # DEBUG: Print original SQL and dataframe info
            print("=== SQL_EXECUTOR DEBUG ===")
            print(f"Original SQL: {sql_query}")
            print(f"DataFrame columns: {list(df.columns)}")
            print(f"DataFrame shape: {df.shape}")
            
            # Step 1: Normalize SQL query with enhanced column mapping
            normalized_sql, column_mapping = self._normalize_sql_query(sql_query, df)
            
            print(f"Normalized SQL: {normalized_sql}")
            print(f"Column mapping: {column_mapping}")
            print("==========================")
            
            # Step 2: Prepare database with cleaned column names
            conn = sqlite3.connect(':memory:')
            df_clean = self._prepare_dataframe_for_sql(df)
            
            # Create table with cleaned names
            df_clean.to_sql(self.table_name, conn, index=False, if_exists='replace')
            
            # Step 3: Execute the normalized SQL
            result_df = pd.read_sql_query(normalized_sql, conn)
            conn.close()
            
            # Step 4: Map column names back to original for display
            result_df = self._restore_column_names(result_df, column_mapping)
            
            return {
                'success': True,
                'data': result_df,
                'row_count': len(result_df),
                'column_count': len(result_df.columns),
                'original_sql': sql_query,
                'executed_sql': normalized_sql,
                'column_mapping': column_mapping
            }
            
        except Exception as e:
            return self._handle_execution_error(sql_query, df, e)
    
    def _normalize_sql_query(self, sql_query: str, df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
        """Normalize SQL query column names - FIXED VERSION that preserves numeric values"""
        
        # Get original and cleaned column names
        original_columns = list(df.columns)
        cleaned_columns = [self._clean_column_name(col) for col in original_columns]
        
        # Create mapping between original and cleaned names
        column_mapping = dict(zip(cleaned_columns, original_columns))
        reverse_mapping = dict(zip(original_columns, cleaned_columns))
        
        normalized_sql = sql_query
        
        # Handle table name variations FIRST
        table_patterns = [
            r'\bFROM\s+["\'`]?(\w+)["\'`]?\b',
            r'\bJOIN\s+["\'`]?(\w+)["\'`]?\b', 
            r'\bUPDATE\s+["\'`]?(\w+)["\'`]?\b',
            r'\bINTO\s+["\'`]?(\w+)["\'`]?\b'
        ]
        for pattern in table_patterns:
            normalized_sql = re.sub(
                pattern, 
                f'FROM {self.table_name}', 
                normalized_sql, 
                flags=re.IGNORECASE
            )
        
        # Extract all column names from SQL
        sql_columns = self._extract_columns_from_sql(normalized_sql)
        
        print(f"📊 Found {len(sql_columns)} column references in SQL: {sql_columns}")
        
        # Replace column names with cleaned versions
        for sql_col in sql_columns:
            # Skip wildcards and aggregate functions
            if sql_col == '*' or sql_col.lower() in ['count', 'sum', 'avg', 'min', 'max', 'count(*)']:
                print(f"  Skipping: {sql_col}")
                continue
            
            # Skip if it's a pure number (values, not columns)
            if sql_col.isdigit():
                print(f"  Skipping numeric value: {sql_col}")
                continue
                
            print(f"  Looking for match for SQL column: '{sql_col}'")
            
            # Find the best matching column from the dataset
            best_match = self._find_best_column_match(sql_col, original_columns, cleaned_columns)
            
            if best_match:
                clean_name = reverse_mapping[best_match]
                print(f"    ✓ Matched to: '{best_match}' -> using cleaned name: '{clean_name}'")
                
                # CRITICAL FIX: Use word boundaries and negative lookahead to avoid matching numbers
                # This ensures we don't replace "id" in "id = 90" and accidentally affect "90"
                patterns = [
                    # Match: 'column_name' but NOT 'column_name123' or '123column_name'
                    (rf'\b{re.escape(sql_col)}\b(?!\d)', clean_name),           # unquoted with word boundary
                    (rf'"{re.escape(sql_col)}"', clean_name),                    # double-quoted
                    (rf"'{re.escape(sql_col)}'", clean_name),                    # single-quoted
                    (rf"`{re.escape(sql_col)}`", clean_name),                    # backtick-quoted
                    (rf'\[{re.escape(sql_col)}\]', clean_name),                  # bracket-quoted
                ]
                
                for pattern, replacement in patterns:
                    if re.search(pattern, normalized_sql, re.IGNORECASE):
                        # Replace but preserve surrounding context
                        normalized_sql = re.sub(
                            pattern, 
                            replacement, 
                            normalized_sql, 
                            flags=re.IGNORECASE
                        )
                        print(f"    Replaced with pattern: {pattern}")
            else:
                print(f"    ✗ No match found for: '{sql_col}'")
                # Try to clean the SQL column name and use it as-is
                cleaned_sql_col = self._clean_column_name(sql_col)
                patterns = [
                    (rf'\b{re.escape(sql_col)}\b(?!\d)', cleaned_sql_col),
                    (rf'"{re.escape(sql_col)}"', cleaned_sql_col),
                    (rf"'{re.escape(sql_col)}'", cleaned_sql_col), 
                    (rf"`{re.escape(sql_col)}`", cleaned_sql_col)
                ]
                for pattern, replacement in patterns:
                    normalized_sql = re.sub(
                        pattern, 
                        replacement, 
                        normalized_sql, 
                        flags=re.IGNORECASE
                    )
        
        # CRITICAL FIX: Handle numeric literals in WHERE clauses MORE CAREFULLY
        # Only remove quotes from COMPLETE numeric values that are clearly standalone
        where_patterns = [
            # Match: WHERE column = '123' -> WHERE column = 123
            # But NOT: WHERE column = '123abc' (keep quotes for alphanumeric)
            # The (?=\s|$|;) ensures we match complete values only
            (r"(WHERE\s+\w+\s*=\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            (r"(WHERE\s+\w+\s*>\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            (r"(WHERE\s+\w+\s*<\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            (r"(WHERE\s+\w+\s*>=\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            (r"(WHERE\s+\w+\s*<=\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            (r"(WHERE\s+\w+\s*!=\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            (r"(AND\s+\w+\s*=\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            (r"(OR\s+\w+\s*=\s*)'(\d+)'(?=\s|$|;|\))", r"\1\2"),
            # Handle IN clauses: IN ('123', '456') -> IN (123, 456)
            (r"IN\s*\(\s*'(\d+)'\s*\)", r"IN (\1)"),
        ]
        
        for pattern, replacement in where_patterns:
            normalized_sql = re.sub(pattern, replacement, normalized_sql, flags=re.IGNORECASE)
        
        # Fix common SQL formatting issues
        # 1. Remove trailing semicolons
        normalized_sql = normalized_sql.rstrip(';').strip()
        
        # 2. Ensure FROM clause uses correct table name
        if not re.search(rf'\bFROM\s+{self.table_name}\b', normalized_sql, re.IGNORECASE):
            from_match = re.search(r'\bFROM\s+(\w+)\b', normalized_sql, re.IGNORECASE)
            if from_match:
                current_table = from_match.group(1)
                normalized_sql = re.sub(
                    rf'\bFROM\s+{current_table}\b', 
                    f'FROM {self.table_name}', 
                    normalized_sql, 
                    flags=re.IGNORECASE
                )
            else:
                # Add FROM clause if missing
                if 'FROM' not in normalized_sql.upper():
                    normalized_sql = normalized_sql + f' FROM {self.table_name}'
        
        # 3. Normalize whitespace (but don't mangle the query)
        normalized_sql = re.sub(r'\s+', ' ', normalized_sql).strip()
        
        # 4. Ensure SELECT keyword is present
        if not re.search(r'^\s*SELECT', normalized_sql, re.IGNORECASE):
            print(f"⚠️ Warning: SQL doesn't start with SELECT: {normalized_sql[:50]}...")
        
        print(f"🔍 Normalized SQL: {normalized_sql}")
        print(f"🗺️ Column mapping: {column_mapping}")
        
        return normalized_sql, column_mapping
    
    def _extract_columns_from_sql(self, sql_query: str) -> List[str]:
        """Extract column names from SQL query - FIXED to avoid numeric confusion"""
        columns = []
        
        # Remove strings and comments to avoid false matches
        # Use a unique placeholder that won't be confused with real data
        clean_sql = re.sub(r"'[^']*'", "'__STR__'", sql_query)
        clean_sql = re.sub(r'"[^"]*"', '"__STR__"', clean_sql)
        clean_sql = re.sub(r'--.*$', '', clean_sql, flags=re.MULTILINE)
        clean_sql = re.sub(r'/\*.*?\*/', '', clean_sql, flags=re.DOTALL)
        
        # Find column names in SELECT clause
        select_match = re.search(
            r'SELECT\s+(.*?)\s+FROM', 
            clean_sql, 
            re.IGNORECASE | re.DOTALL
        )
        if select_match:
            select_clause = select_match.group(1)
            parts = [part.strip() for part in select_clause.split(',')]
            for part in parts:
                # Remove functions and keep column name
                col_name = re.sub(r'^\w+\(([^)]+)\)', r'\1', part)
                col_name = re.sub(r'\s+AS\s+\w+', '', col_name, flags=re.IGNORECASE)
                col_name = col_name.strip()
                # CRITICAL: Skip pure numbers and wildcards
                if col_name and col_name != '*' and not col_name.isdigit() and col_name != '__STR__':
                    columns.append(col_name)
        
        # Find column names in WHERE clause
        where_match = re.search(
            r'WHERE\s+(.*?)(?:\s+GROUP BY|\s+ORDER BY|\s+HAVING|\s+LIMIT|$)', 
            clean_sql, 
            re.IGNORECASE | re.DOTALL
        )
        if where_match:
            where_clause = where_match.group(1)
            # Extract column names BEFORE operators (left side of comparison)
            # This avoids capturing the VALUES which are on the right side
            where_cols = re.findall(
                r'\b(\w+)\s*(?:=|!=|>|<|>=|<=|LIKE|IN|IS|BETWEEN)', 
                where_clause
            )
            # Filter out numeric strings and SQL keywords
            sql_keywords = {'and', 'or', 'not', 'in', 'between', 'like', 'is'}
            where_cols = [
                col for col in where_cols 
                if not col.isdigit() and col.lower() not in sql_keywords
            ]
            columns.extend(where_cols)
        
        # Find column names in GROUP BY clause
        group_match = re.search(
            r'GROUP BY\s+(.*?)(?:\s+ORDER BY|\s+HAVING|\s+LIMIT|$)', 
            clean_sql, 
            re.IGNORECASE | re.DOTALL
        )
        if group_match:
            group_clause = group_match.group(1)
            group_cols = [col.strip() for col in group_clause.split(',')]
            group_cols = [col for col in group_cols if not col.isdigit()]
            columns.extend(group_cols)
        
        # Find column names in ORDER BY clause
        order_match = re.search(
            r'ORDER BY\s+(.*?)(?:\s+LIMIT|$)', 
            clean_sql, 
            re.IGNORECASE | re.DOTALL
        )
        if order_match:
            order_clause = order_match.group(1)
            order_cols = [col.strip() for col in order_clause.split(',')]
            # Remove ASC/DESC
            order_cols = [
                re.sub(r'\s+(ASC|DESC)$', '', col, flags=re.IGNORECASE) 
                for col in order_cols
            ]
            order_cols = [col for col in order_cols if not col.isdigit()]
            columns.extend(order_cols)
        
        # Remove duplicates
        return list(set(columns))
    
    def _find_best_column_match(self, sql_column: str, original_columns: List[str], cleaned_columns: List[str]) -> str:
        """Find the best matching column from available columns"""
        # Remove quotes and normalize
        clean_sql_col = self._clean_column_name(sql_column)
        
        # Strategy 1: Exact match with cleaned names
        for orig_col, clean_col in zip(original_columns, cleaned_columns):
            if clean_sql_col == clean_col:
                return orig_col
        
        # Strategy 2: Case-insensitive exact match
        for orig_col, clean_col in zip(original_columns, cleaned_columns):
            if clean_sql_col.lower() == clean_col.lower() or clean_sql_col.lower() == orig_col.lower():
                return orig_col
        
        # Strategy 3: Partial match
        for orig_col, clean_col in zip(original_columns, cleaned_columns):
            if clean_sql_col in clean_col or clean_col in clean_sql_col:
                return orig_col
        
        # Strategy 4: Word boundary match (e.g., "customer_id" matches "customerid")
        clean_sql_col_no_underscore = clean_sql_col.replace('_', '')
        for orig_col, clean_col in zip(original_columns, cleaned_columns):
            clean_col_no_underscore = clean_col.replace('_', '')
            if clean_sql_col_no_underscore == clean_col_no_underscore:
                return orig_col
        
        # Strategy 5: Fuzzy matching
        try:
            matches = get_close_matches(clean_sql_col, cleaned_columns, n=1, cutoff=0.6)
            if matches:
                match_index = cleaned_columns.index(matches[0])
                return original_columns[match_index]
        except ImportError:
            pass
        
        # Strategy 6: Synonym matching
        synonym_map = {
            'date': ['timestamp', 'time', 'datetime', 'dt'],
            'name': ['title', 'label', 'identifier', 'desc', 'description'],
            'value': ['amount', 'quantity', 'number', 'total', 'sum', 'price', 'cost'],
            'id': ['identifier', 'code', 'number', 'num'],
            'category': ['type', 'class', 'group', 'kind'],
            'status': ['state', 'condition'],
            'age': ['years', 'year'],
            'salary': ['wage', 'income', 'pay'],
            'gender': ['sex'],
            'country': ['nation', 'location'],
            'city': ['town', 'location']
        }
        
        # Check if SQL column is a key in synonym map
        for key, synonyms in synonym_map.items():
            if key in clean_sql_col:
                for orig_col, clean_col in zip(original_columns, cleaned_columns):
                    if any(synonym in clean_col for synonym in synonyms):
                        return orig_col
        
        # Check if any synonym is in SQL column
        for orig_col, clean_col in zip(original_columns, cleaned_columns):
            for key, synonyms in synonym_map.items():
                if any(synonym in clean_sql_col for synonym in [key] + synonyms):
                    if any(synonym in clean_col for synonym in [key] + synonyms):
                        return orig_col
        
        return None
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column name for SQL compatibility"""
        # Remove quotes first
        cleaned = re.sub(r'["\'`]', '', str(column_name))
        # Replace spaces, special characters with underscores
        cleaned = re.sub(r'[^\w]', '_', cleaned)
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Ensure it starts with a letter
        if cleaned and not cleaned[0].isalpha():
            cleaned = 'col_' + cleaned
        return cleaned.lower()
    
    def _prepare_dataframe_for_sql(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe with SQL-compatible column names"""
        df_clean = df.copy()
        df_clean.columns = [self._clean_column_name(col) for col in df_clean.columns]
        return df_clean
    
    def _restore_column_names(self, result_df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Restore original column names in results"""
        result_df_restored = result_df.copy()
        result_df_restored.columns = [
            column_mapping.get(col, col) for col in result_df.columns
        ]
        return result_df_restored
    
    def _handle_execution_error(self, sql_query: str, df: pd.DataFrame, error: Exception) -> Dict[str, Any]:
        """Handle SQL execution errors with detailed diagnostics"""
        available_columns = list(df.columns)
        cleaned_columns = [self._clean_column_name(col) for col in available_columns]
        
        error_info = {
            'success': False,
            'error': str(error),
            'error_type': type(error).__name__,
            'available_columns': available_columns,
            'cleaned_columns': cleaned_columns,
            'sql_query': sql_query,
            'suggestions': self._generate_error_suggestions(sql_query, df, str(error))
        }
        
        return error_info
    
    def _generate_error_suggestions(self, sql_query: str, df: pd.DataFrame, error_msg: str) -> List[str]:
        """Generate helpful suggestions based on error type"""
        suggestions = []
        available_columns = list(df.columns)
        cleaned_columns = [self._clean_column_name(col) for col in available_columns]
        
        # Column-related errors
        if 'no such column' in error_msg.lower():
            column_match = re.search(r"no such column: ([^\s]+)", error_msg, re.IGNORECASE)
            if column_match:
                missing_col = column_match.group(1)
                suggestions.append(
                    f"Column '{missing_col}' not found. Available columns: {', '.join(available_columns)}"
                )
        
        # Table-related errors
        elif 'no such table' in error_msg.lower():
            suggestions.append(f"Use table name '{self.table_name}' in your query")
        
        # Syntax errors
        elif 'syntax error' in error_msg.lower():
            suggestions.append(
                "Check SQL syntax. Basic format: SELECT column1, column2 FROM data WHERE condition"
            )
        
        # General suggestion
        if not suggestions:
            suggestions.append(f"Available columns: {', '.join(available_columns)}")
            suggestions.append(f"Table name should be: {self.table_name}")
        
        return suggestions