import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PostgresConfig(BaseModel):
    """Configuration for Postgres"""
    url: Optional[str] = Field(None, description="Database name for Postgres")
    user: Optional[str] = Field(None, description="User for Postgres")
    password: Optional[str] = Field(None, description="Password for Postgres")
    host: Optional[str] = Field(None, description="Host for Postgres")
    port: Optional[int] = Field(None, description="Port for Postgres")
    max_connections: Optional[int] = Field(None, description="Maximum connections for Postgres")
    socket_timeout: Optional[int] = Field(None, description="Socket timeout for Postgres")
    
    @validator("url")
    def validate_url(cls, value: Optional[str]) -> Optional[str]:
        """Validate URL"""
        if value is None:
            raise ValueError("Database name is required")
        return value
    
    @validator("user")
    def validate_user(cls, value: Optional[str]) -> Optional[str]:
        """Validate user"""
        if value is None:
            raise ValueError("User is required")
        return value
    
    @validator("password")
    def validate_password(cls, value: Optional[str]) -> Optional[str]:
        """Validate password"""
        if value is None:
            raise ValueError("Password is required")
        return value
    
    @validator("host")
    def validate_host(cls, value: Optional[str]) -> Optional[str]:
        """Validate host"""
        if value is None:
            raise ValueError("Host is required")
        return value
        
    @validator("port")
    def validate_port(cls, value: Optional[int]) -> Optional[int]:
        """Validate port"""
        if value is None:
            raise ValueError("Port is required")
        return value
    
    @validator("max_connections")
    def validate_max_connections(cls, value: Optional[int]) -> Optional[int]:
        """Validate max connections"""
        if value is None:
            raise ValueError("Max connections is required")
        return value
    
    @validator("socket_timeout")
    def validate_socket_timeout(cls, value: Optional[int]) -> Optional[int]:
        """Validate socket timeout"""
        if value is None:
            raise ValueError("Socket timeout is required")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.model_dump()
    
    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Load configuration from environment variables"""
        return cls(
            url=os.environ.get("POSTGRES_DB"),
            user=os.environ.get("POSTGRES_USER"),
            password=os.environ.get("POSTGRES_PASSWORD"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            max_connections=int(os.environ.get("POSTGRES_MAX_CONNECTIONS", "10")),
            socket_timeout=int(os.environ.get("POSTGRES_TIMEOUT", "30"))
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PostgresConfig":
        """Load configuration from dictionary"""
        return cls(
            url=config_dict.get("POSTGRES_DB"),
            user=config_dict.get("POSTGRES_USER"),
            password=config_dict.get("POSTGRES_PASSWORD"),
            host=config_dict.get("POSTGRES_HOST", "localhost"),
            port=int(config_dict.get("POSTGRES_PORT", "5432")),
            max_connections=int(config_dict.get("POSTGRES_MAX_CONNECTIONS", "10")),
            socket_timeout=int(config_dict.get("POSTGRES_TIMEOUT", "30"))
        )

    def initialize(self):
        """Initialize configuration"""
        if any(v is None for v in [self.url, self.user, self.password, self.host, self.port]):
            raise ValueError("All required configuration values must be set")
        # Additional initialization logic can be added here
        return True
