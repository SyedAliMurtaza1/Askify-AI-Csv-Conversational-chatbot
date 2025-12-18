import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CSVChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Chunk DataFrame into meaningful text chunks for RAG"""
        chunks = []
        
        # 1. Schema information chunk
        schema_chunk = self._create_schema_chunk(df)
        chunks.append(schema_chunk)
        
        # 2. Statistical summary chunks
        stats_chunks = self._create_statistical_chunks(df)
        chunks.extend(stats_chunks)
        
        # 3. Sample data chunks
        sample_chunks = self._create_sample_chunks(df)
        chunks.extend(sample_chunks)
        
        # 4. Column-specific chunks (for large datasets)
        column_chunks = self._create_column_chunks(df)
        chunks.extend(column_chunks)
        
        return chunks
    
    def _create_schema_chunk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create schema information chunk"""
        schema_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict()
        }
        
        content = f"SCHEMA INFORMATION:\n"
        content += f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns\n"
        content += "Columns and their data types:\n"
        for col, dtype in schema_info["dtypes"].items():
            missing = schema_info["missing_values"][col]
            content += f"  - {col}: {dtype} ({missing} missing values)\n"
        
        return {
            "content": content,
            "metadata": {"type": "schema", "chunk_id": "schema_0"}
        }
    
    def _create_statistical_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create statistical summary chunks"""
        chunks = []
        
        # Numerical columns statistics
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            stats = df[num_cols].describe()
            content = "NUMERICAL COLUMNS STATISTICS:\n"
            for col in num_cols:
                content += f"\n{col}:\n"
                content += f"  Mean: {stats[col]['mean']:.2f}\n"
                content += f"  Std: {stats[col]['std']:.2f}\n"
                content += f"  Min: {stats[col]['min']:.2f}\n"
                content += f"  Max: {stats[col]['max']:.2f}\n"
            
            chunks.append({
                "content": content,
                "metadata": {"type": "statistics", "chunk_id": "stats_numerical"}
            })
        
        # Categorical columns statistics
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            content = f"CATEGORICAL COLUMN: {col}\n"
            content += f"Top values and their counts:\n"
            for value, count in value_counts.items():
                content += f"  - {value}: {count}\n"
            
            chunks.append({
                "content": content,
                "metadata": {"type": "statistics", "column": col, "chunk_id": f"stats_{col}"}
            })
        
        return chunks
    
    def _create_sample_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create sample data chunks that help SQL generation"""
        chunks = []
        
        # 1. First, show the schema clearly
        content = "DATASET SCHEMA AND SAMPLE VALUES:\n"
        content += "=" * 50 + "\n\n"
        
        # Show column info with sample values
        for col in df.columns:
            content += f"Column: {col}\n"
            content += f"  Type: {df[col].dtype}\n"
            
            # Show unique values for categorical/non-numeric columns
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                unique_vals = df[col].dropna().unique()[:8]  # First 8 unique values
                if len(unique_vals) > 0:
                    content += f"  Sample values: {', '.join(map(str, unique_vals))}\n"
            else:
                # For numeric columns, show range
                if pd.api.types.is_numeric_dtype(df[col]):
                    content += f"  Range: {df[col].min():.2f} to {df[col].max():.2f}\n"
                    # Show common values if they exist
                    common_vals = df[col].value_counts().head(3).index.tolist()
                    if common_vals:
                        content += f"  Common values: {', '.join(map(str, common_vals))}\n"
            
            content += "\n"
        
        chunks.append({
            "content": content,
            "metadata": {"type": "schema_samples", "chunk_id": "schema_with_samples"}
        })
        
        # 2. Then show actual data rows in a clean format
        sample_rows = min(5, len(df))
        table_content = f"DATA ROWS (First {sample_rows} rows):\n"
        table_content += "=" * 50 + "\n\n"
        
        # Create a simple table
        columns_to_show = list(df.columns)[:8]  # Show first 8 columns to avoid overload
        
        # Header
        table_content += " | ".join([f"{col:<15}"[:15] for col in columns_to_show]) + "\n"
        table_content += "-" * (len(columns_to_show) * 16) + "\n"
        
        # Rows
        for idx, row in df.head(sample_rows).iterrows():
            row_values = []
            for col in columns_to_show:
                val = str(row[col])
                if len(val) > 12:
                    val = val[:9] + "..."
                row_values.append(f"{val:<15}"[:15])
            table_content += " | ".join(row_values) + "\n"
        
        chunks.append({
            "content": table_content,
            "metadata": {"type": "data_rows", "chunk_id": "data_rows_table"}
        })
        
        # 3. Add a chunk with query-focused information
        query_focused = "QUERY-FOCUSED DATA INSIGHTS:\n"
        query_focused += "=" * 50 + "\n\n"
        
        # Identify columns that look like IDs or keys
        id_columns = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['id', 'code', 'number', 'no'])]
        if id_columns:
            query_focused += f"ID/Key columns: {', '.join(id_columns)}\n\n"
            
            # Show sample values for ID columns
            for col in id_columns[:3]:  # Show first 3 ID columns
                sample_vals = df[col].dropna().unique()[:5]
                query_focused += f"  {col} samples: {', '.join(map(str, sample_vals))}\n"
            query_focused += "\n"
        
        # Identify numeric columns that might be filtered on
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            query_focused += f"Numeric filter columns: {', '.join(numeric_cols)}\n\n"
            
            # Show range for each numeric column
            for col in numeric_cols[:5]:  # Show first 5 numeric columns
                if len(df[col].dropna()) > 0:
                    query_focused += f"  {col}: {df[col].min():.0f} to {df[col].max():.0f} "
                    query_focused += f"(avg: {df[col].mean():.0f})\n"
        
        # Identify text columns that might be filtered on
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            query_focused += f"\nText filter columns: {', '.join(text_cols[:5])}\n"
            # Show most common values for first text column
            if text_cols:
                col = text_cols[0]
                common_vals = df[col].value_counts().head(3).index.tolist()
                query_focused += f"  Common '{col}' values: {', '.join(common_vals)}\n"
        
        chunks.append({
            "content": query_focused,
            "metadata": {"type": "query_insights", "chunk_id": "query_focused"}
        })
        
        return chunks
    def _create_column_chunks(self, df: pd.DataFrame):
        """Create column chunks WITHOUT hardcoded examples"""
        chunks = []
        
        for col in df.columns:
            content = f"Column: {col}\n"
            
            if pd.api.types.is_numeric_dtype(df[col]):
                content += "Type: Numeric\n"
                content += "Use for: Calculations (SUM, AVG), Comparisons (>, <), Aggregation\n"
                
                if len(df[col].dropna()) > 0:
                    content += f"Range: Values from {df[col].min():.0f} to {df[col].max():.0f}\n"
                    if df[col].nunique() < 20:
                        content += "Contains: Distinct numeric values\n"
            
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                content += "Type: Text\n"
                content += "Use for: Text matching (LIKE), Grouping, Filtering\n"
                
                if len(df[col].dropna()) > 0:
                    sample = str(df[col].iloc[0])
                    
                    # Analyze format dynamically without hardcoding
                    if '-' in sample and len(sample.split('-')) == 3:
                        parts = sample.split('-')
                        if len(parts[0]) == 2 and len(parts[2]) == 2:
                            content += "Format: DD-XXX-YY (date-like)\n"
                        elif len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2:
                            content += "Format: YYYY-MM-DD (date-like)\n"
                    elif '@' in sample and '.' in sample:
                        content += "Format: Contains @ and . (email-like)\n"
                    elif sample.isdigit():
                        content += "Format: Numeric string\n"
                    else:
                        # Count words to give generic description
                        word_count = len(sample.split())
                        if word_count == 1:
                            content += "Format: Single word\n"
                        else:
                            content += "Format: Multi-word text\n"
            
            # Describe relationship to other columns dynamically
            content += f"\nUnique values: {df[col].nunique()}\n"
            content += f"Missing values: {df[col].isnull().sum()}\n"
            
            # Calculate statistics dynamically
            if pd.api.types.is_numeric_dtype(df[col]) and len(df[col].dropna()) > 10:
                q25 = df[col].quantile(0.25)
                q75 = df[col].quantile(0.75)
                content += f"Typical range (25-75%): {q25:.0f} to {q75:.0f}\n"
            
            chunks.append({
                "content": content,
                "metadata": {"type": "column", "column": col}
            })
        
        return chunks
    
    def _create_dynamic_join_guidance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create JOIN guidance based on actual columns in the data"""
        content = "QUERY PATTERNS FOR FINDING MATCHING RECORDS:\n"
        content += "=" * 50 + "\n\n"
        
        # Dynamically identify columns that could be used for joins
        potential_id_cols = []
        potential_group_cols = []
        potential_match_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Identify potential ID columns (high uniqueness)
            if df[col].nunique() == len(df[col].dropna()):  # All values unique
                potential_id_cols.append(col)
            
            # Identify potential grouping columns (low/medium uniqueness)
            elif df[col].nunique() < 20:
                potential_group_cols.append(col)
            
            # Identify date-like columns
            if 'date' in col_lower or 'time' in col_lower:
                potential_match_cols.append(col)
            elif df[col].dtype == 'object' and len(df[col].dropna()) > 0:
                sample = str(df[col].iloc[0])
                if '-' in sample and len(sample.split('-')) == 3:
                    potential_match_cols.append(col)
        
        if potential_id_cols:
            content += f"Unique identifier columns: {', '.join(potential_id_cols[:3])}\n"
            content += "Use these for JOIN conditions: ON a.id = b.id\n\n"
        
        if potential_group_cols:
            content += f"Grouping columns (few unique values): {', '.join(potential_group_cols[:5])}\n"
            content += "Use these in GROUP BY or JOIN grouping conditions\n\n"
        
        if potential_match_cols:
            content += f"Matching columns (for finding duplicates): {', '.join(potential_match_cols[:5])}\n"
            content += "Use these in self-JOIN ON conditions: ON a.column = b.column\n\n"
        
        # Show generic patterns using column types, not names
        content += "GENERIC PATTERNS (adapt to actual column names):\n\n"
        
        content += "1. Find records with same values:\n"
        content += "   SELECT a.*, b.*\n"
        content += "   FROM data a\n"
        content += "   INNER JOIN data b ON a.match_column = b.match_column\n"
        content += "   WHERE a.id_column != b.id_column\n\n"
        
        content += "2. Count duplicates by group:\n"
        content += "   SELECT group_column, match_column, COUNT(*) as count\n"
        content += "   FROM data\n"
        content += "   GROUP BY group_column, match_column\n"
        content += "   HAVING COUNT(*) > 1\n\n"
        
        content += "3. Find matching records within groups:\n"
        content += "   SELECT a.*, b.*\n"
        content += "   FROM data a\n"
        content += "   INNER JOIN data b ON a.group_column = b.group_column\n"
        content += "       AND a.match_column = b.match_column\n"
        content += "       AND a.id_column < b.id_column\n\n"
        
        content += "⚠️ Use actual column names from the schema above.\n"
        content += "⚠️ Do NOT hardcode example values like '21-JUN-07'.\n"
        content += "⚠️ Let the query find matches dynamically using column comparisons.\n"
        
        return {
            "content": content,
            "metadata": {"type": "join_patterns", "chunk_id": "dynamic_join_guide"}
        }            