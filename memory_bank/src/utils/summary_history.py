from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
import openai

from config.llm_config import LLMConfig

class SummaryHistory:
    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig.from_env()
        self.config = config
        
        # Initialize OpenAI client directly
        self.openai_client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        self.prompt_template = (
            "You are a helpful assistant that summarizes conversations between a user and a validator agent. "
            "Your task is to create a concise summary that captures the main intent and key information "
            "from the conversation until the validator confirmed the user's request is clear (pass=True). "
            "Focus on the user's final clarified intent and important context.\n\n"
            "Please summarize this conversation history:\n{input}\n\n"
            "Summary should be concise but capture the essence of what the user wants."
        )

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history into a readable string"""
        formatted_lines = []
        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            formatted_lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_lines)

    def summarize(self, input: List[Dict]) -> str:
        """
        Summarize conversation history
        
        Args:
            input (List[Dict]): List of conversation turns with 'role' and 'content' keys
            
        Returns:
            str: Summary of the conversation
        """
        try:
            formatted_input = self._format_conversation_history(input)
            
            # Create prompt with input
            prompt = self.prompt_template.format(input=formatted_input)
            
            # Call OpenAI API directly
            response = self.openai_client.chat.completions.create(
                model=self.config.model_name or "gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content
            return summary.strip() if summary else "No summary generated"
        except Exception as e:
            # Fallback to simple concatenation if LLM fails
            return f"Conversation summary: {len(input)} turns of conversation between user and validator agent."