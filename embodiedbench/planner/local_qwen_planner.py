# import torch
# import json
# from PIL import Image
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from peft import PeftModel


# class LocalQwenPlanner:
#     def __init__(
#         self,
#         base_model_id,
#         lora_path,
#         language_skill_set,
#         system_prompt,
#         examples,
#         n_shot=0,
#         device="cuda"
#     ):
#         self.device = device
#         self.language_skill_set = language_skill_set
#         self.system_prompt = system_prompt
#         self.examples = examples
#         self.n_shot = n_shot
#         self.planner_steps = 0
#         self.output_json_error = 0

#         print("Loading base model...")
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             base_model_id,
#             dtype=torch.bfloat16,
#             device_map="auto"
#         )

#         print("Loading LoRA adapter...")
#         self.model = PeftModel.from_pretrained(self.model, lora_path)
#         self.model.eval()

#         self.processor = AutoProcessor.from_pretrained(base_model_id)

#     def reset(self):
#         self.planner_steps = 0

#     def set_actions(self, language_skill_set):
#         self.language_skill_set = language_skill_set

#     def update_info(self, info):
#         pass

#     def build_messages(self, image_path, instruction):
#         return [
#             {
#                 "role": "system",
#                 "content": [{"type": "text", "text": self.system_prompt}]
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image_path},
#                     {"type": "text", "text": f"Human Instruction: {instruction}"}
#                 ]
#             }
#         ]

#     def act(self, image_path, instruction):
#         self.planner_steps += 1

#         image = Image.open(image_path).convert("RGB")
#         messages = self.build_messages(image_path, instruction)

#         chat = self.processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inputs = self.processor(
#             text=chat,
#             images=image,
#             return_tensors="pt"
#         ).to(self.device)

#         with torch.no_grad():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=512,
#                 do_sample=False
#             )

#         response = self.processor.decode(
#             output_ids[0],
#             skip_special_tokens=True
#         )

#         try:
#             # Extract JSON from model output
#             json_start = response.find("{")
#             json_end = response.rfind("}") + 1
#             parsed = json.loads(response[json_start:json_end])

#             action_id = parsed["executable_plan"][0]["action_id"]
#             reasoning = parsed.get("reasoning_and_reflection", "")

#         except Exception:
#             self.output_json_error += 1
#             return -1, "JSON parsing error"

#         return action_id, reasoning


import torch
import json
from PIL import Image
# 1. FIXED: Swapped to Qwen2VL to match your training script
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor 
from peft import PeftModel


class LocalQwenPlanner:
    def __init__(
        self,
        base_model_id,
        lora_path,
        language_skill_set,
        system_prompt,
        examples,
        n_shot=0,
        device="cuda"
    ):
        self.device = device
        self.language_skill_set = language_skill_set
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.planner_steps = 0
        self.output_json_error = 0

        print("Loading base model...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16, # Added torch_dtype parameter
            device_map="auto"
        )

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(base_model_id)

    def reset(self):
        self.planner_steps = 0

    def set_actions(self, language_skill_set):
        self.language_skill_set = language_skill_set

    def update_info(self, info):
        pass

    def build_messages(self, image_path, instruction):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Human Instruction: {instruction}"}
                ]
            }
        ]

    def act(self, image_path, instruction):
        self.planner_steps += 1

        image = Image.open(image_path).convert("RGB")
        messages = self.build_messages(image_path, instruction)

        chat = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. FIXED: Wrapped chat and image in lists to ensure safe batch dimensionality
        inputs = self.processor(
            text=[chat],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        # 3. FIXED: Trim out the input prompt so we only decode the model's new answer
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        try:
            # Safely extract JSON from the trimmed model output
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            parsed = json.loads(response[json_start:json_end])

            action_id = parsed["executable_plan"][0]["action_id"]
            reasoning = parsed.get("reasoning_and_reflection", "")

        except Exception as e:
            self.output_json_error += 1
            return -1, f"JSON parsing error: {str(e)}\nRaw Response: {response}"

        return action_id, reasoning