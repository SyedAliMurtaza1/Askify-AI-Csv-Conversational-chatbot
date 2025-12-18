from typing import List, Dict, Any
import re

class QuestionAnalyzer:
    def __init__(self):
        self.aggregation_patterns = {
            'average': ['average', 'avg', 'mean'],
            'sum': ['sum', 'total'],
            'count': ['count', 'number of', 'how many'],
            'max': ['max', 'highest', 'top', 'most'],
            'min': ['min', 'lowest', 'bottom', 'least']
        }
    
    def preprocess_question(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower().strip()
        
        columns = self._extract_column_mentions(question)
        aggregation = self._identify_aggregation(question_lower)
        filters = self._extract_filters(question)
        keywords = self._extract_keywords(question_lower)
        question_type = self._detect_question_type(question_lower)
        
        if aggregation == 'count' and filters:
            question_type = 'filtered_count'
        
        return {
            'original_question': question,
            'normalized_question': question_lower,
            'columns': columns,
            'aggregation': aggregation,
            'filters': filters,
            'keywords': keywords,
            'question_type': question_type
        }
    
    def _extract_column_mentions(self, question: str) -> List[str]:
        found = []
        
        patterns = [
            r'(\b[\w]+\b)\s*[=<>!]',  # column=value
            r'(\b[\w]+\b)\s+(?:is|are|was|were)\s+',  # column is value
            r'(\b[\w]+\b)\s+(?:has|have)\s+',  # column has value
            r'(\b[\w]+\b)\s+(?:greater|less|more|above|below)',  # column greater than
            r'count\s+of\s+(\b[\w]+\b)',  # count of column
            r'sum\s+of\s+(\b[\w]+\b)',  # sum of column
            r'avg\s+of\s+(\b[\w]+\b)',  # avg of column
            r'top\s+\d+\s+(\b[\w]+\b)',  # top 5 column
            r'by\s+(\b[\w]+\b)',  # group by column
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            found.extend([m.strip() for m in matches])
        
        return list(set(found))
    
    def _identify_aggregation(self, question: str) -> str:
        for agg_type, patterns in self.aggregation_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', question):
                    return agg_type
        return 'none'
    
    def _extract_filters(self, question: str) -> List[Dict]:
        filters = []
        
        patterns = [
            (r'(\b[\w]+\b)\s*=\s*["\']([^"\']+)["\']', '='),  # col="value"
            (r'(\b[\w]+\b)\s*=\s*(\b[\w]+\b)', '='),  # col=value
            (r'(\b[\w]+\b)\s*=\s*(\d+)', '='),  # col=123
            (r'(\b[\w]+\b)\s*>\s*(\d+)', '>'),  # col>123
            (r'(\b[\w]+\b)\s*<\s*(\d+)', '<'),  # col<123
            (r'(\b[\w]+\b)\s+is\s+["\']([^"\']+)["\']', '='),  # col is "value"
            (r'(\b[\w]+\b)\s+is\s+(\b[\w]+\b)', '='),  # col is value
        ]
        
        for pattern, operator in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    column = match[0].strip()
                    value = match[1]
                    
                    filters.append({
                        'column': column.upper(),
                        'operator': operator,
                        'value': value
                    })
        
        return filters
    
    def _extract_keywords(self, question: str) -> List[str]:
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
                    'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
                    'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how'}
        
        words = re.findall(r'\b[a-z]{2,}\b', question.lower())
        keywords = [w for w in words if w not in stopwords]
        
        return keywords
    
    def _detect_question_type(self, question: str) -> str:
        if re.search(r'\b(count|number|how many)\b', question, re.IGNORECASE):
            return 'count'
        if re.search(r'\b(sum|total)\b', question, re.IGNORECASE):
            return 'sum'
        if re.search(r'\b(avg|average|mean)\b', question, re.IGNORECASE):
            return 'average'
        if re.search(r'\b(top|highest|max)\b', question, re.IGNORECASE):
            return 'top'
        if re.search(r'\b(where|with|having)\b', question, re.IGNORECASE):
            return 'filter'
        return 'general'
    
    def get_enhanced_search_terms(self, question: str) -> List[str]:
        analysis = self.preprocess_question(question)
        queries = [question]
        
        queries.append(' '.join(analysis['keywords']))
        
        for col in analysis['columns'][:3]:
            queries.append(col)
        
        for filter_cond in analysis['filters']:
            col = filter_cond.get('column', '')
            val = filter_cond.get('value', '')
            if col and val:
                queries.append(f"{col} {val}")
                queries.append(val)
        
        if analysis['aggregation'] != 'none':
            queries.append(analysis['aggregation'])
        
        return list(set(queries))


class ContextBuilder:
    def __init__(self, max_context_length: int = 2000):
        self.max_context_length = max_context_length
        self.question_analyzer = QuestionAnalyzer()
    
    def build_context(self, retrieved_chunks: List[Dict[str, Any]], user_question: str) -> str:
        if not retrieved_chunks:
            return "No data found."
        
        context_parts = [f"Question: {user_question}"]
        
        schema_chunks = [c for c in retrieved_chunks if c.get("metadata", {}).get("type") == "schema"]
        if schema_chunks:
            context_parts.append("Schema:")
            context_parts.append(schema_chunks[0]["content"])
        
        column_chunks = [c for c in retrieved_chunks if c.get("metadata", {}).get("type") == "column"]
        if column_chunks:
            context_parts.append("Columns:")
            for chunk in column_chunks[:5]:
                context_parts.append(chunk["content"])
        
        sample_chunks = [c for c in retrieved_chunks if c.get("metadata", {}).get("type") == "samples"]
        if sample_chunks:
            context_parts.append("Sample Data:")
            context_parts.append(sample_chunks[0]["content"])
        
        context = '\n'.join(context_parts)
        
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length]
        
        return context
    
    def get_enhanced_search_terms(self, question: str) -> List[str]:
    
        terms = []
        
        # 1. Original question
        clean_q = re.sub(r'\s+', ' ', question).strip()
        terms.append(clean_q)
        
        # 2. Words from question (3+ letters)
        words = re.findall(r'\b\w{3,}\b', clean_q, re.IGNORECASE)
        for word in words:
            word_low = word.lower()
            # Skip very common words
            if word_low not in {'give','the', 'and', 'for', 'with', 'have', 'from'}:
                terms.append(word)
                terms.append(word.upper())
        
        # 3. Extract patterns like X = Y
        patterns = [
            r'(\w+)\s*[=:]\s*[\'"]?(\w+)[\'"]?',
            r'(\w+\s+\w+)\s*[=:]\s*[\'"]?(\w+)[\'"]?',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, clean_q, re.IGNORECASE)
            for left, right in matches:
                # Clean left side (potential column)
                left_clean = re.sub(r'\s+', '_', left.strip()).upper()
                if left_clean:
                    terms.append(left_clean)
                
                # Right side (value)
                if right:
                    terms.append(right.upper())
        
        # 4. Remove duplicates and empty
        final = []
        seen = set()
        for term in terms:
            if term and term not in seen:
                seen.add(term)
                final.append(term)
        
        return final