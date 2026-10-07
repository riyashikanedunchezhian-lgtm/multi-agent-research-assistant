"""LLM client with token tracking and model selection."""

import os
from typing import Optional, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()


class LLMClient:
    """Client for interacting with LLMs with token tracking."""

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            if model is None:
                model = os.getenv("ROUTER_MODEL", "gpt-4o-mini")

            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
            )
        
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
            if model is None:
                model = "claude-3-haiku-20240307"
            
            self.llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def invoke(self, prompt: str) -> tuple[str, dict]:
        """
        Invoke the LLM and return the response with token usage.
        
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        response = self.llm.invoke(prompt)
        
        # Extract token usage
        token_usage = {
            "prompt_tokens": response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") else 0,
            "completion_tokens": response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") else 0,
            "total_tokens": response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0,
        }
        
        return response.content, token_usage

    def get_model_name(self) -> str:
        """Get the model name being used."""
        return self.model if self.model else self.llm.model_name
