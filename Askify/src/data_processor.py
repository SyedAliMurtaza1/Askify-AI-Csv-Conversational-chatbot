import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.df = None
        self.schema = None
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load and clean CSV data"""
        try:
            self.df = pd.read_csv(file_path)
            self._clean_data()
            self._infer_schema()
            return self.df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def _clean_data(self):
        """Basic data cleaning"""
        # Remove completely empty columns
        self.df = self.df.dropna(axis=1, how='all')
        
        # Fill partial missing values
        for col in self.df.columns:
            if self.df[col].dtype in ['object']:
                self.df[col] = self.df[col].fillna('Unknown')
            else:
                self.df[col] = self.df[col].fillna(0)
    
    def _infer_schema(self):
        """Infer schema information"""
        self.schema = {
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'shape': self.df.shape,
            'sample_data': self.df.head(3).to_dict('records')
        }
    
    def get_schema_info(self) -> str:
        """Get schema information as string for LLM context"""
        if not self.schema:
            return "No data loaded"
        
        schema_str = f"Dataset Shape: {self.schema['shape']}\nColumns:\n"
        for col, dtype in self.schema['dtypes'].items():
            schema_str += f"  - {col} ({dtype})\n"
        
        return schema_str
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        if self.df is None:
            return {}
        
        stats = {
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'total_missing': self.df.isnull().sum().sum(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        return stats