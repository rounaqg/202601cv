from ollama import Client
from typing import List, Callable, Dict, Any

class LocalStrandAgent:
    def __init__(self, model: str, tools: List[Callable], system_prompt: str):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        
        # Explicitly connect to localhost to ensure model loads in RAM
        self.client = Client(host='http://localhost:11434')
        
        # Create a mapping for tool execution
        self.tool_map = {tool.__name__: tool for tool in self.tools}

    def chat(self, user_query: str):
        self.messages.append({"role": "user", "content": user_query})

        while True:
            # 1. THINK (Using explicit client to force local load)
            print(f"🤔 Thinking with {self.model} on localhost...")
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=self.tools, 
            )
            
            message = response['message']
            self.messages.append(message)

            # 2. PLAN & ACT
            if message.get('tool_calls'):
                for tool in message['tool_calls']:
                    function_name = tool['function']['name']
                    arguments = tool['function']['arguments']
                    
                    if function_name in self.tool_map:
                        print(f"🛠️ Tool Call: {function_name}")
                        function_to_call = self.tool_map[function_name]
                        result = function_to_call(**arguments)
                        
                        # 3. REFLECT
                        self.messages.append({
                            "role": "tool",
                            "content": str(result),
                        })
            else:
                # 4. FINAL ANSWER
                return message['content']

    def update_model(self, new_model: str):
        self.model = new_model
        
    def clear_history(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]