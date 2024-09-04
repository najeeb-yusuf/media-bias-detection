from .base_models import BaseModel
from octoai.client import OctoAI
from octoai.text_gen import ChatMessage
import time

class OctAI(BaseModel):
    def __init__(self, model_name, api_key):
        super().__init__(api_key=api_key, model_name=model_name)
        self.client = OctoAI(api_key=api_key)
        self.product_name = 'octoai'

    def predict_single(self, text, prompt, fine_tune=False, no_labels=2, temperature=1.5):
        result = self.client.text_gen.create_chat_completion_stream(
            max_tokens=4,
            messages=[
                ChatMessage(
                    content=prompt,
                    role="system"
                ),
                ChatMessage(
                    content=text,
                    role="user"
                )
            ],
            model=self.model_name,
            presence_penalty=0,
            temperature=temperature
        )
        response = [i.choices[0].delta.content if i.choices[0].delta.content else '' for i in result]
        return ''.join(response)
