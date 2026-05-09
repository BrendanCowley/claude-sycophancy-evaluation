from anthropic import Anthropic
import datetime
from dotenv import load_dotenv
import os


# creds
load_dotenv()

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
client = Anthropic(api_key=CLAUDE_API_KEY)


class Claude_Conversation:
    def __init__(self, model_id: str):
        self.messages = [] # conversation history
        self.model = model_id
        self.temperature = 0 # don't want random responses, this is the default value anyways but want to be explicit about temp being 0
        self.response_history = [] # LLM response history
    
    def send(self, message: str, client) -> dict:
        self.messages.append({"role": "user", "content": message})
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=self.messages
        )
        
        assistant_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_text})

        # creating response object with appropriate metadata
        full_response = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "response_text": response.content[0].text,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.response_history.append(full_response)
        
        return full_response
    
    def reset(self): # this really shouldn't be used as new conversations should be a new instance, but just in case
        self.messages = []


if __name__ == "__main__":
    #example usage
    model_id = "claude-haiku-4-5" # using most lightweight model for testing code
    message = "Hello, Claude"

    conversation = Claude_Conversation(model_id=model_id)
    client = Anthropic(api_key=CLAUDE_API_KEY) # we can define this here as we are only planning on using 1 client

    response = conversation.send(message, client)

    print(response)
    print(conversation.messages)
