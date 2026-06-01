import asyncio
import json
import httpx
import time
import re
import os
from agentskg.agents.base import AgentConfig, BaseAgent
from agentskg.prompts.verify_fact_prompt import SEMANTIC_BOOL_PROMPT


class SemanticAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(timeout=30.0)

    def _parse_llm_response(self, response_text: str) -> list[int] | None:
        """
        Parse the feature list from the LLM's text response.
        Expected format: <result>[0,1,0,1,0,0,0]</result>
        or directly [0,1,0,1,0,0,0]
        or even possibly 0,1,0,1,0,0,0 (as a more relaxed alternative)
        """
        if not response_text:
            return None
        parsed_list_str = None
        xml_match = re.search(r"<result>\[(.*?)\]</result>", response_text, re.DOTALL)
        if xml_match:
            parsed_list_str = xml_match.group(1)
        else:
            direct_list_match = re.search(r"\[(.*?)\]", response_text)
            if direct_list_match:
                parsed_list_str = direct_list_match.group(1)
            else:
                comma_separated_match = re.fullmatch(
                    r"\s*([01](?:\s*,\s*[01])*(?:\s*,)?)\s*", response_text
                )
                if comma_separated_match:
                    parsed_list_str = comma_separated_match.group(1)

        if parsed_list_str is not None:
            try:
                if parsed_list_str.endswith(","):
                    parsed_list_str = parsed_list_str[:-1]

                str_values = parsed_list_str.split(",")
                int_values = []
                for s_val in str_values:
                    s_val_cleaned = s_val.strip()
                    if s_val_cleaned.isdigit() and s_val_cleaned in ["0", "1"]:
                        int_values.append(int(s_val_cleaned))
                    else:
                        return None

                if len(int_values) == 7:
                    return int_values
                else:
                    return None
            except ValueError as e:
                return None
            except Exception as e:
                return None
        else:
            return None

    async def execute(self, context: str) -> list:
        api_url = os.getenv("API_URL")
        api_key = os.getenv("API_KEY")
        if not api_url or not api_key:
            print("Error: SemanticAgent failed to load API_URL or API_KEY.")
            return None

        api_url = f"{api_url.rstrip('/')}/chat/completions"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": SEMANTIC_BOOL_PROMPT + context}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = await self.client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            content = (
                response_data.get("choices", [{}])[0].get("message", {}).get("content")
            )
            if not content:
                print("SemanticAgent - Error: 'content' missing in LLM response.")
                return None

            parsed_list = self._parse_llm_response(content)
            return parsed_list

        except httpx.HTTPStatusError as e:
            print(
                f"SemanticAgent - HTTP error: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            print(f"SemanticAgent - An unexpected error occurred: {e}")
            return None

    async def validate(self):
        pass

    async def close_client(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            print("SemanticAgent's HTTP client closed.")
