# algorithm_agent.py
from llm_utils import call_llm
import pandas as pd

class AlgorithmAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_code(self, task: str, df: pd.DataFrame) -> str:
        # Get detailed dataset information
        column_info = self._get_column_info(df)
        dtypes_info = self._get_dtypes_info(df)
        sample_data = df.head(3).to_string()
        
        prompt = (
            "You are an algorithm specialist. You specialize in tasks involving computational problem-solving and "
            "algorithm development.\n\n"
            
            "CRITICAL INSTRUCTIONS:\n"
            "1. **ONLY** reply with Python code (no prose, no markdown backticks, no explanations)\n"
            "2. Define a function exactly named `solution(df)` that accepts a pandas DataFrame and returns a result\n"
            "3. Use the EXACT column names provided - DO NOT rename or modify column names\n"
            "4. If column names have spaces or special characters, use bracket notation: df['column name']\n"
            "5. Do not read or write files. Do not ask for input\n"
            "6. Keep the function self-contained and working with the provided DataFrame structure\n"
            "7. Return serializable results (DataFrame, dict, list, number, string)\n"
            "8. Handle potential missing values appropriately\n\n"
            
            "DATASET INFORMATION:\n"
            f"Columns: {column_info}\n"
            f"Data types: {dtypes_info}\n"
            f"Shape: {df.shape}\n\n"
            
            "SAMPLE DATA (first 3 rows):\n"
            f"{sample_data}\n\n"
            
            f"TASK: {task}\n\n"
            
            "Remember: Use EXACT column names as shown above. Do NOT rename columns unless explicitly required by the task."
        )
        
        code = self._call_llm(prompt)
        return str(code)
    
    def _get_column_info(self, df: pd.DataFrame) -> str:
        """Get formatted column names."""
        return ", ".join([f"'{col}'" for col in df.columns])
    
    def _get_dtypes_info(self, df: pd.DataFrame) -> str:
        """Get data types for each column."""
        dtype_list = [f"'{col}': {df[col].dtype}" for col in df.columns]
        return "{" + ", ".join(dtype_list) + "}"
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt."""
        response = self.llm.invoke(prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            code = response.content
        else:
            code = str(response)
        
        # Clean up the response - remove markdown backticks if present
        code = code.strip()
        if code.startswith('```'):
            # Remove opening backticks
            lines = code.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove closing backticks
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            code = '\n'.join(lines)
        
        return code.strip()
