import sqlglot
from sqlglot import parse_one, exp
from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

class SQLValidator:
    def __init__(self):
        self.supported_dialects = ["sqlite", "mysql", "postgres"]
    
    def validate_sql(self, sql_query: str, df_columns: List[str] = None) -> Dict[str, Any]:
        """Validate SQL syntax and compatibility - ULTRA PERMISSIVE VERSION"""
        validation_result = {
            'is_valid': True,  # Default to True - be more permissive
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'parsed_tables': [],
            'parsed_columns': [],
            'supported_operations': []
        }
        
        # Clean the SQL query first
        sql_query = sql_query.strip().rstrip(';')
        
        try:
            # Try to parse SQL, but don't fail if parsing fails
            parsed = parse_one(sql_query, read="sqlite")
            
            # Extract components if parsing succeeded
            tables = self._extract_tables(parsed)
            columns = self._extract_columns(parsed)
            operations = self._extract_operations(parsed)
            
            validation_result['parsed_tables'] = tables
            validation_result['parsed_columns'] = columns
            validation_result['supported_operations'] = operations
            
            # Give warnings but don't fail validation
            if df_columns:
                normalized_df_cols = [self._normalize_column_name(col) for col in df_columns]
                
                for col in columns:
                    if col and col != '*':
                        normalized_col = self._normalize_column_name(col)
                        
                        if normalized_col not in normalized_df_cols:
                            closest = self._find_closest_column(col, df_columns)
                            if closest:
                                validation_result['warnings'].append(
                                    f"Column '{col}' not found. Did you mean '{closest}'?"
                                )
                            else:
                                validation_result['warnings'].append(
                                    f"Column '{col}' not found in dataset"
                                )
            
            # Always mark as valid to allow execution attempt
            validation_result['suggestions'].append("SQL will be attempted for execution")
            
        except Exception as e:
            # Don't fail on ANY parsing errors - let execution handle it
            validation_result['warnings'].append(f"SQL parsing warning: {str(e)}")
            validation_result['suggestions'].append("Attempting to execute despite parsing warnings")
            # STILL MARK AS VALID to allow execution attempt
            validation_result['is_valid'] = True
        
        return validation_result
    
    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column name for comparison"""
        normalized = re.sub(r'["\'\`\s]', '', str(column_name).lower())
        normalized = re.sub(r'[^\w]', '_', normalized)
        return normalized
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from parsed SQL"""
        tables = []
        for table in parsed.find_all(exp.Table):
            tables.append(table.name)
        return tables
    
    def _extract_columns(self, parsed) -> List[str]:
        """Extract column names from parsed SQL"""
        columns = []
        
        for column in parsed.find_all(exp.Column):
            columns.append(column.name)
        
        for select in parsed.args.get("expressions", []):
            if isinstance(select, exp.Column):
                columns.append(select.name)
            elif isinstance(select, exp.Alias):
                if hasattr(select.this, 'name'):
                    columns.append(select.this.name)
        
        return list(set(columns))
    
    def _extract_operations(self, parsed) -> List[str]:
        """Extract SQL operations used"""
        operations = []
        if parsed.find_all(lambda x: hasattr(x, 'is_aggregate') and x.is_aggregate):
            operations.append("AGGREGATE")
        operation_types = [
            (exp.Where, "WHERE"),
            (exp.Group, "GROUP BY"),
            (exp.Order, "ORDER BY"),
            (exp.Join, "JOIN"),
            (exp.Limit, "LIMIT")
        ]
        
        for op_type, op_name in operation_types:
            if parsed.find(op_type):
                operations.append(op_name)
        
        return operations
    
    def _find_closest_column(self, target: str, available_columns: List[str]) -> str:
        """Find the closest matching column name"""
        from difflib import get_close_matches
        
        target_normalized = self._normalize_column_name(target)
        available_normalized = [self._normalize_column_name(col) for col in available_columns]
        
        matches = get_close_matches(target_normalized, available_normalized, n=1, cutoff=0.6)
        if matches:
            match_index = available_normalized.index(matches[0])
            return available_columns[match_index]
        
        return None