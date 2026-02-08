import ollama
from typing import List, Callable

class LocalStrandAgent:
    def __init__(self, model: str, tools: List[Callable], system_prompt: str):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.client = ollama.Client(host='http://localhost:11434')
        self.tool_map = {tool.__name__: tool for tool in self.tools}

    def chat(self, user_query: str):
        self.messages.append({"role": "user", "content": user_query})

        while True:
            # 1. THINK
            print(f"🤔 {self.model} is thinking...")
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=self.tools, 
            )
            
            message = response['message']
            self.messages.append(message)

            # 2. ACT
            if message.get('tool_calls'):
                for tool in message['tool_calls']:
                    function_name = tool['function']['name']
                    arguments = tool['function']['arguments']
                    
                    if function_name in self.tool_map:
                        print(f"🛠️ Executing Tool: {function_name}")
                        try:
                            result = self.tool_map[function_name](**arguments)
                        except Exception as e:
                            result = f"Tool Error: {e}"
                        
                        self.messages.append({
                            "role": "tool",
                            "content": str(result),
                        })
            else:
                return message['content']

    def update_model(self, new_model: str):
        self.model = new_model
        
    def clear_history(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]