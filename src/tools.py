"""Tools for the agent to call when retrieval isn't enough."""

import math
import re
from typing import Optional
from urllib.parse import quote

import httpx
from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Allow only safe mathematical operations
        allowed_chars = set("0123456789+-*/.() %^")
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "Error: Expression contains invalid characters"

        # Replace ^ with ** for Python
        expression = expression.replace("^", "**")

        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, {})

        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web for information using a free search API.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 3)
    
    Returns:
        A formatted string with search results
    """
    try:
        # Using DuckDuckGo's instant answer API (free, no API key needed)
        url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json"
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

        # Extract relevant information
        results = []
        
        # Abstract (if available)
        if data.get("Abstract"):
            results.append(f"Abstract: {data['Abstract']}")
        
        # AbstractText (alternative)
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        
        # AbstractSource
        if data.get("AbstractSource"):
            results.append(f"Source: {data['AbstractSource']}")
        
        # Definition (if available)
        if data.get("Definition"):
            results.append(f"Definition: {data['Definition']}")
        
        # Related topics
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(f"- {topic['Text']}")
        
        if not results:
            return f"No results found for query: {query}"
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def get_current_date() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current date and time as a string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def code_interpreter(code: str) -> str:
    """
    Execute Python code safely and return the result.
    
    Args:
        code: Python code to execute
    
    Returns:
        Output from the code execution or error message
    """
    try:
        # Restrict what can be executed
        allowed_modules = {"math", "random", "statistics", "datetime"}
        
        # Check for dangerous imports
        import_pattern = re.compile(r"^import\s+|^from\s+\w+\s+import")
        if import_pattern.search(code):
            # Check if it's an allowed module
            for line in code.split("\n"):
                if line.strip().startswith(("import ", "from ")):
                    module_name = line.split()[1].split(".")[0]
                    if module_name not in allowed_modules:
                        return f"Error: Module '{module_name}' is not allowed"
        
        # Create a safe execution environment
        safe_globals = {
            "__builtins__": {},
            "math": math,
        }
        
        # Execute the code
        exec(code, safe_globals)
        
        # Try to get the result of the last expression
        result = safe_globals.get("_result", "Code executed successfully")
        
        return str(result)
        
    except Exception as e:
        return f"Error executing code: {str(e)}"


# Tool registry
AVAILABLE_TOOLS = [calculator, web_search, get_current_date, code_interpreter]

TOOL_MAP = {tool.name: tool for tool in AVAILABLE_TOOLS}


def get_tool_by_name(name: str) -> Optional[object]:
    """Get a tool by its name."""
    return TOOL_MAP.get(name)


def list_tools() -> list[str]:
    """List all available tool names."""
    return list(TOOL_MAP.keys())
