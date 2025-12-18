class PromptEngineer:
    def __init__(self):
        self.system_prompt = """You are a SQL generator for table 'data'.

RULES:
1. Use EXACT column names from schema
2. Use EXACT numbers from question - NEVER change them
3. Format: SELECT ... FROM data WHERE ...
4. Text: 'quotes', Numbers: no quotes
5. Return ONLY the SQL query
6. For "same X" questions, use the column X in GROUP BY or JOIN ON condition

EXAMPLES:
Question: show where units_sold = 8082
SQL: SELECT * FROM data WHERE units_sold = 8082

Question: salary 5000 and department_id 90
SQL: SELECT * FROM data WHERE salary = 5000 AND department_id = 90

Question: count employees by department
SQL: SELECT department_id, COUNT(*) FROM data GROUP BY department_id

AVOID:
✗ Changing numbers (8082→8889)
✗ Missing FROM clause
✗ Wrong column names

CORRECT PATTERNS:
- ✓ SELECT * FROM [table] WHERE [column] = value
- ✓ SELECT COUNT(*) FROM [table] WHERE [column] = 'value'
- ✓ SELECT [column1], COUNT(*) FROM [table] GROUP BY [column1]
- ✓ Always preserve exact numeric values from the question

SQL STRUCTURE (MANDATORY ORDER):
SELECT [what to select]
FROM [table from schema]
WHERE [conditions if needed]
GROUP BY [if aggregating]
ORDER BY [if sorting]
[LIMIT number]             # Optional - ONLY if limiting requested
"""
    
    def create_prompt(self, context: str, question: str) -> str:
        """Create the final prompt for the LLM with enhanced instructions"""
        
        # Extract any numeric values from the question
        import re
        numeric_values = re.findall(r'\b\d+\b', question)
        
        question_lower = question.lower()
        
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"{'='*60}\n"
        prompt += f"DATASET INFORMATION\n"
        prompt += f"{'='*60}\n\n"
        prompt += f"{context}\n\n"
        prompt += f"{'='*60}\n"
        prompt += f"USER QUESTION\n"
        prompt += f"{'='*60}\n\n"
        prompt += f"{question}\n\n"
        
        # Check for grouping patterns
        grouping_terms = ['group by', 'grouped by', 'by', 'per', 'each', 'count by', 'sum by', 'average by']
        is_grouping_question = any(term in question_lower for term in grouping_terms)
        
        if re.search(r'\b(top|first|limit to|show only)\s+\d+\b', question_lower):
            prompt += "NOTE: Question asks for specific number of results - use LIMIT\n\n"
        elif re.search(r'\b(highest|lowest|max|min)\b', question_lower):
            prompt += "NOTE: Question asks for highest/lowest - use ORDER BY\n\n"
        elif is_grouping_question:
            prompt += "NOTE: This is a grouping question - use GROUP BY, NO ORDER BY or LIMIT needed\n\n"
        
        # DYNAMIC JOIN GUIDANCE SECTION (ADDED)
        # Check for patterns without hardcoding specific words
        has_same_pattern = ('same' in question_lower and 
                        any(word in question_lower for word in ['date', 'time', 'day', 'value', 'record']))
        
        has_find_pattern = ('find' in question_lower and 
                        any(word in question_lower for word in ['duplicate', 'match', 'similar', 'pair']))
        
        has_group_pattern = 'group' in question_lower or 'by' in question_lower or 'each' in question_lower
        
    
        
        if numeric_values:
            prompt += f"CRITICAL: The user mentioned these EXACT numbers: {', '.join(numeric_values)}\n"
            prompt += f"You MUST use these exact values in your SQL query. DO NOT change them.\n\n"
        
        prompt += f"{'='*60}\n"
        prompt += f"YOUR SQL QUERY\n"
        prompt += f"{'='*60}\n\n"
        prompt += "Generate a SQL query that answers the question using:\n"
        prompt += "- The exact column names from the schema above\n"
        prompt += "- Table name 'data'\n"
        prompt += "- Proper SQL syntax: SELECT ... FROM data WHERE ...\n"
        
        if numeric_values:
            prompt += f"- The EXACT numeric values: {', '.join(numeric_values)} (DO NOT CHANGE THESE)\n"
        
        # Add guidance based on question type
        if is_grouping_question:
            prompt += "- For grouping questions: SELECT column, COUNT(*) or SUM(column) FROM data GROUP BY column\n"
        
        # ADDITIONAL JOIN GUIDANCE IF DETECTED
        if has_same_pattern or has_find_pattern:
            prompt += "- For matching/same value questions: Use GROUP BY with HAVING COUNT(*) > 1 or self-JOIN\n"
        
        prompt += "\nQuery: "
        
        return prompt
    def needs_joins_adapter(self, question: str) -> bool:
        """Simple detection for JOIN queries - add this method"""
        question_lower = question.lower()
        
        join_keywords = [
            'join', 'together', 'both', 'and their', 'with their',
            'combine', 'including', 'along with', 'showing both' ,
            'with'
        ]
        
        return any(keyword in question_lower for keyword in join_keywords)
    def extract_sql_from_response(self, response: str) -> str:
        """Extract clean SQL query from model response"""
        import re
        
        # Remove common prefixes
        response = response.strip()
        
        # Remove SQL: prefix if present
        if response.upper().startswith("SQL:"):
            response = response[4:].strip()
        
        # Remove Query: prefix if present
        if response.upper().startswith("QUERY:"):
            response = response[6:].strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split('\n')
            # Remove first line (```) and last line (```)
            if len(lines) > 2:
                response = '\n'.join(lines[1:-1])
        
        # Remove sql/SQL language identifier
        if response.lower().startswith("sql\n"):
            response = response[4:].strip()
        
        # Remove trailing semicolon
        response = response.rstrip(';').strip()
        
        # Remove comments
        lines = response.split('\n')
        clean_lines = []
        for line in lines:
            # Remove inline comments
            if '--' in line:
                line = line.split('--')[0]
            if '/*' in line:
                line = line.split('/*')[0]
            line = line.strip()
            if line:
                clean_lines.append(line)
        
        # Join lines and clean up whitespace
        response = ' '.join(clean_lines)
        
        # Ensure it starts with SELECT
        if not response.upper().startswith('SELECT'):
            # Try to find SELECT in the response
            select_pos = response.upper().find('SELECT')
            if select_pos != -1:
                response = response[select_pos:]
        
        return response.strip()
    
    def validate_and_clean_sql(self, sql_query: str) -> str:
        """Validate and clean SQL query before execution"""
        import re
        
        sql_query = self.extract_sql_from_response(sql_query)
        
        # FIX: Ensure FROM clause is before WHERE clause
        if 'WHERE' in sql_query.upper() and 'FROM' in sql_query.upper():
            # Check if WHERE comes before FROM (wrong order)
            where_pos = sql_query.upper().find('WHERE')
            from_pos = sql_query.upper().find('FROM')
            
            if where_pos < from_pos:
                # Extract parts
                select_match = re.search(r'(SELECT\s+.*?)\s+WHERE', sql_query, re.IGNORECASE)
                where_match = re.search(r'WHERE\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
                from_match = re.search(r'FROM\s+(.*?)(?:\s|$)', sql_query, re.IGNORECASE)
                
                if select_match and where_match and from_match:
                    select_part = select_match.group(1)
                    where_part = where_match.group(1)
                    from_part = from_match.group(1)
                    
                    # Rebuild in correct order
                    sql_query = f"{select_part} FROM {from_part} WHERE {where_part}"
        
        # FIX: Add FROM clause if missing
        if 'FROM' not in sql_query.upper() and 'WHERE' in sql_query.upper():
            # Insert FROM data before WHERE
            sql_query = re.sub(
                r'\s+WHERE\s+',
                ' FROM data WHERE ',
                sql_query,
                flags=re.IGNORECASE
            )
        
        # Ensure single line (for display)
        sql_query = ' '.join(sql_query.split())
        
        return sql_query