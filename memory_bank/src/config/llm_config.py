import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for LLM API"""
    api_key: Optional[str] = Field(None, description="API key for LLM API")
    base_url: Optional[str] = Field(None, description="Base URL for LLM API")
    model_name: Optional[str] = Field(None, description="Model name for LLM API")
    
    @validator("api_key")
    def validate_api_key(cls, value: Optional[str]) -> Optional[str]:
        """Validate API key"""
        if value is None:
            raise ValueError("API key is required")
        return value
    
    @validator("base_url")
    def validate_base_url(cls, value: Optional[str]) -> Optional[str]:
        """Validate base URL"""
        if value is None:
            raise ValueError("Base URL is required")
        return value
    
    @validator("model_name")
    def validate_model_name(cls, value: Optional[str]) -> Optional[str]:
        """Validate model name"""
        if value is None:
            raise ValueError("Model name is required")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.model_dump()
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables"""
        return cls(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL"),
            model_name=os.environ.get("MODEL_NAME")
        )