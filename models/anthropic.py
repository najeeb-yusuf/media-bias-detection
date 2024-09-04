from .base_models import BaseModel
import logging

import anthropic

# ANTHRIPIC_API_KEY = os.environ('ANTHROPIC_API_KEY')

class ClaudeAI(BaseModel):
    def __init__(self, api_key, model_name):
        super().__init__(api_key=api_key, model_name=model_name)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.product_name = 'claudeai'

    def predict_single(self, text, prompt, fine_tuned=False, no_labels=2):
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0,
                system=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            }
                        ]
                    }
                ]
            )
            return message.content[0].text

        except Exception as e:
            logging.error(e)
            return 'none'
