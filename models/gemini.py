from .base_models import BaseModel
import google.generativeai as genai
class Gemini(BaseModel):
    def __init__(self, api_key, model_name):
        super().__init__(api_key=api_key, model_name=model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.product_name = 'gemini'

    def predict_single(self, text, prompt, fine_tune=False, no_labels=2, temperature=0.25):
        result = self.model.generate_content(
            prompt + ':' + text,
            generation_config=genai.GenerationConfig(
                max_output_tokens=3,
                temperature=temperature
            )
        )
        try:
            return result.text
        except Exception as e:
            return ''
