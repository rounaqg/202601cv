import ollama
from openai import OpenAI
import json
from typing import List, Callable

class LocalStrandAgent:
    def __init__(self, provider: str, model: str, tools: List[Callable], system_prompt: str):
        self.provider = provider
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        
        # Initialize clients
        if self.provider == "Ollama":
            self.client = ollama.Client(host='http://localhost:11434')
        else: # LM Studio
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def chat(self, user_query: str):
        self.messages.append({"role": "user", "content": user_query})
        while True:
            print(f"🤔 {self.model} via {self.provider} is thinking...")
            
            if self.provider == "Ollama":
                response = self.client.chat(model=self.model, messages=self.messages, tools=self.tools)
                msg = response['message']
                tool_calls = msg.get('tool_calls')
            else:
                # OpenAI-compatible format for LM Studio
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=[self._gen_tool_schema(t) for t in self.tools] if self.tools else None
                )
                msg = response.choices[0].message
                tool_calls = msg.tool_calls

            self.messages.append(msg)

            if tool_calls:
                for tc in tool_calls:
                    # Normalize tool call data across providers
                    f_name = tc['function']['name'] if self.provider == "Ollama" else tc.function.name
                    f_args = tc['function']['arguments'] if self.provider == "Ollama" else json.loads(tc.function.arguments)
                    
                    if f_name in self.tool_map:
                        print(f"🛠️ Executing Tool: {f_name}")
                        res = str(self.tool_map[f_name](**f_args))
                        
                        tool_msg = {"role": "tool", "content": res}
                        if self.provider == "LM Studio":
                            tool_msg.update({"tool_call_id": tc.id, "name": f_name})
                        self.messages.append(tool_msg)
            else:
                return msg['content'] if self.provider == "Ollama" else msg.content

    def _gen_tool_schema(self, func):
        """Helper to create OpenAI-compatible tool schemas for LM Studio."""
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "No description",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"]
                }
            }
        }

    def update_config(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        if provider == "Ollama":
            self.client = ollama.Client(host='http://localhost:11434')
        else:
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")