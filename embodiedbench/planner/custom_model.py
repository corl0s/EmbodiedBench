import requests
import torch
import os
import io
import requests

temperature = 0
max_completion_tokens = 2048
server_url = os.environ.get('server_url')

class CustomModel():
    def __init__(self, model_path, language_only):
        self.model_path = model_path
        self.language_only = language_only
        self.model_type = 'custom'
        

    def respond(self, prompt, obs=None):        
        with open(obs, "rb") as img_file:
            files = {"image": img_file}
            data = {"sentence": prompt}
            response = requests.post(server_url, files=files, data=data)

        res= response.json()['response']
        if response.status_code != 200:
            print("Error:", response.text)
        return res
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from PIL import Image
# import torch

# class CustomModel():

#     def __init__(self, model_path, language_only):
#         self.language_only = language_only

#         print("Loading model from:", model_path)
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )

#         self.processor = AutoProcessor.from_pretrained(model_path)

#     def respond(self, prompt, obs=None):

#         image = Image.open(obs).convert("RGB")

#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": prompt},
#             ],
#         }]

#         text = self.processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inputs = self.processor(
#             text=[text],
#             images=[image],
#             return_tensors="pt"
#         ).to(self.model.device)

#         output = self.model.generate(
#             **inputs,
#             max_new_tokens=256
#         )

#         response = self.processor.batch_decode(
#             output,
#             skip_special_tokens=True
#         )[0]

#         return response

