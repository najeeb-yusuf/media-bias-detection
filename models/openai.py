from .base_models import BaseModel
from openai import OpenAI
class ChatGPTPrompt(BaseModel):
    def __init__(self, model_name, api_key):
        super().__init__(model_name=model_name, api_key=api_key)
        self.client = OpenAI(api_key=api_key)
        self.product_name = 'chatgpt'
    def predict_single(self, text, prompt, fine_tuned=False, no_labels=2, temperature=0.8):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=temperature
        )
        return completion.choices[0].message.content
