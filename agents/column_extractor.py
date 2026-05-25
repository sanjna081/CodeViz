# column_extractor.py
import re
from typing import List, Tuple, Set
from difflib import get_close_matches

class ColumnExtractor:
    def __init__(self, df_columns: List[str]):
        """
        df_columns: List of actual column names from the DataFrame
        """
        self.df_columns = df_columns
        self.df_columns_lower = [col.lower() for col in df_columns]
    
    def extract_columns(self, query: str, threshold: float = 0.8) -> Tuple[List[str], List[str]]:
        """
        Extract column names from user query using hybrid approach.
        
        Returns:
            (exact_matches, fuzzy_matches)
        """
        exact_matches = []
        fuzzy_candidates = []
        
        # 1. EXACT MATCHING (case-insensitive)
        query_lower = query.lower()
        for col in self.df_columns:
            col_lower = col.lower()
            # Check if column appears as whole word in query
            pattern = r'\b' + re.escape(col_lower) + r'\b'
            if re.search(pattern, query_lower):
                exact_matches.append(col)
        
        # 2. QUOTED TEXT MATCHING
        # Match anything in quotes that might be column names
        quoted_patterns = re.findall(r'["\']([^"\']+)["\']', query)
        for quoted in quoted_patterns:
            matches = get_close_matches(quoted.lower(), self.df_columns_lower, n=1, cutoff=threshold)
            if matches:
                idx = self.df_columns_lower.index(matches[0])
                col = self.df_columns[idx]
                if col not in exact_matches:
                    fuzzy_candidates.append(col)
        
        # 3. FUZZY MATCHING for potential column references
        # Extract potential column-like tokens (capitalized words, snake_case, etc.)
        tokens = self._extract_potential_columns(query)
        for token in tokens:
            if token.lower() not in [m.lower() for m in exact_matches]:
                matches = get_close_matches(token.lower(), self.df_columns_lower, n=1, cutoff=threshold)
                if matches:
                    idx = self.df_columns_lower.index(matches[0])
                    col = self.df_columns[idx]
                    if col not in exact_matches and col not in fuzzy_candidates:
                        fuzzy_candidates.append(col)
        
        return exact_matches, fuzzy_candidates
    
    def _extract_potential_columns(self, query: str) -> List[str]:
        """
        Extract tokens that look like column names:
        - CamelCase words
        - snake_case words
        - Words with numbers
        - Capitalized words
        """
        tokens = set()
        
        # Snake_case or words with underscores
        tokens.update(re.findall(r'\b\w+_\w+\b', query))
        
        # CamelCase (at least one lowercase followed by uppercase)
        tokens.update(re.findall(r'\b[a-z]+[A-Z]\w*\b', query))
        
        # Words with numbers
        tokens.update(re.findall(r'\b\w*\d+\w*\b', query))
        
        # Capitalized words (but not sentence starters)
        # This is tricky - only add if it's NOT at start of sentence
        words = query.split()
        for i, word in enumerate(words):
            # Skip if it's the first word or follows sentence-ending punctuation
            if i > 0 and words[i-1][-1] not in '.!?':
                if word[0].isupper() and word not in ['I', 'A']:
                    tokens.add(word.strip('.,!?;:'))
        
        return list(tokens)
    
    def extract_with_llm_verification(self, query: str, llm) -> List[str]:
        """
        Parse columns first, then verify with LLM.
        This gives you the best of both worlds.
        """
        exact, fuzzy = self.extract_columns(query)
        
        candidates = exact + fuzzy
        
        if not candidates:
            # No parsing matches - fall back to LLM
            return self._llm_extract_fallback(query, llm)
        
        # If we have candidates, verify with LLM
        return self._verify_with_llm(query, candidates, llm)
    
    def _verify_with_llm(self, query: str, candidates: List[str], llm) -> List[str]:
        """
        Ask LLM to verify which candidates are actually referenced in the query.
        """
        from llm_utils import call_llm
        
        prompt = f"""Given this user query and candidate column names, identify which columns are ACTUALLY referenced in the query.

Query: "{query}"

Candidate columns: {', '.join(candidates)}

RULES:
1. Only select columns that are EXPLICITLY mentioned or clearly implied in the query
2. If query says "between X and Y", return both X and Y
3. If query is vague like "analyze the data", return empty list
4. Return ONLY the column names, one per line, nothing else

Selected columns:"""
        
        response = call_llm(llm, prompt)
        
        # Parse LLM response
        selected = []
        for line in response.strip().split('\n'):
            line = line.strip().strip('"-,;')
            if line in candidates:
                selected.append(line)
        
        return selected if selected else candidates  # Fallback to candidates if LLM fails
    
    def _llm_extract_fallback(self, query: str, llm) -> List[str]:
        """
        Pure LLM extraction when parsing finds nothing.
        """
        from llm_utils import call_llm
        
        prompt = f"""Extract column names mentioned in this query. Choose ONLY from this list of available columns:

Available columns: {', '.join(self.df_columns)}

Query: "{query}"

RULES:
1. Return ONLY column names that are explicitly mentioned in the query
2. Use EXACT column names from the available list
3. One column per line
4. If no specific columns mentioned, return "ALL"

Column names:"""
        
        response = call_llm(llm, prompt)
        
        if "ALL" in response.upper():
            return []
        
        extracted = []
        for line in response.strip().split('\n'):
            line = line.strip().strip('"-,;')
            if line in self.df_columns:
                extracted.append(line)
        
        return extracted