import asyncio
import json
import os
import httpx
from typing import Optional
from agentskg.agents.base import BaseAgent, AgentConfig


class ExtractorAgent(BaseAgent):
    """
    Agent responsible for calling the LLM API (SiliconFlow) to extract information
    based on a given prompt and role. Uses httpx for async requests.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(timeout=120.0)

    async def execute(self, context: str, agent_role: str) -> Optional[str]:
        """
        Executes the LLM call to extract information.

        Args:
            context: The prompt content for the user role.
            agent_role: The content for the system role (defining the agent's persona/task).

        Returns:
            The content string from the LLM response, or None if an error occurs.
        """
        api_url = os.getenv("API_URL")
        api_key = os.getenv("API_KEY")
        if not api_url or not api_key:
            print(
                "Error: ExtractorAgent failed to load API_URL or API_KEY from environment variables."
            )
            print("Please check your .env file.")
            return None
        api_url = f"{api_url.rstrip('/')}/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": agent_role},
                {"role": "user", "content": context},
            ],
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if hasattr(self.config, "enable_thinking") and self.config.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            response = await self.client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content")
                if content:
                    return content
                else:
                    print(
                        f"Error: 'content' missing in LLM response message: {message}"
                    )
                    return None
            else:
                print(
                    f"Error: 'choices' missing or empty in LLM response: {response_data}"
                )
                return None

        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error occurred: {e}. Response text: {response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during LLM call: {e}")
            return None

    async def validate(self, result: Optional[str]) -> bool:
        """
        Validates the raw output from the LLM.
        (Placeholder - implement specific validation if needed)
        """
        return isinstance(result, str) and bool(result.strip())

    async def close(self):
        """Closes the underlying httpx client."""
        await self.client.aclose()
        print("ExtractorAgent's HTTP client closed.")
