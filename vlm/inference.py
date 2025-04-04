import os
os.environ["HF_HOME"] = "checkpoints/"
os.environ["TRITON_CACHE_DIR"] = "/capstor/scratch/cscs/mhasan/"

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import re

model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
video_dir = "/capstor/store/cscs/swissai/a03/mariam/DrivingVQA/GEM_filtered_test/n_frames_25_steps_50_sampler_EulerEDMSamplerDynamicPyramid_guider_VanillaCFG_cfg_scale_min_1.0_cfg_scale_max_1.5_cond_aug_0.0_sigma_max_150.0_rho_7/virtual/videos/"
json_path = "/capstor/store/cscs/swissai/a03/mariam/DrivingVQA/splits/transformed_test_data _filtered_human.json"
output_json_path = "/capstor/store/cscs/swissai/a03/mariam/DrivingVQA/outputs.json"

# Load the processor
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# Load the model with device mapping and appropriate data type
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    device_map="balanced",
    attn_implementation="flash_attention_2",
)


with open(json_path, 'r') as f:
    qa_data = json.load(f)

filename_to_qas = {}
for entry_list in qa_data.values():
    for item in entry_list:
        filename_to_qas.setdefault(item["img_filename"], []).append(item)
all_outputs = []
for fname in os.listdir(video_dir):
    parts = fname.split("_")
    if len(parts) < 3:
        continue
    mid_id = parts[1]
    img_key = f"{mid_id}.jpg"
    if img_key not in filename_to_qas:
        print(f"Skipping {fname} — no match in JSON for {img_key}")
        continue

    video_path = os.path.abspath(os.path.join(video_dir, fname))
    video_uri = f"file://{video_path}"
    qa_items = filename_to_qas[img_key]

    for qa in qa_items:
        question = qa["question"]
        explanation = qa["interleaved_explanation"]

        prompt = (
            f"you are a driving expert, you will be presented with a video and a question, it answer, and explanation for that answer.\n"
            f"the questions and answers are mainly about the ego car driving in the video.\n"
            f"Your job is to check whether the ego car driving in the video follows the behavior described in the explanation. Mainly focus on the behavior explained in the explanation\n"
            f"if there is an expected behavior described in the explaation about another car in the scene, also penalize it\n"
            f"at the end of your answer, you should give a score in the following format: Final Score 0 or 0.5 or 1.\n"
            f"0 means the ego car is not acting correctly, 0.5 means the ego car is acting correctly but another car in the scene is not acting correctly. You should be very strict\n" 
            f"if the ego behaviour is wrong but the other cars are acting correctly, you should give a score of 0.\n"
            "1 means everything is realistic and follows the explanation provided as well as the social norms.\n"
            "if the ego car should stop and it doesn't not stop right away, give it a 0.\n"
            "if the traffic red is light, the ego car should not be moving untill the color changed. The car should be stopped before the traffic light, it should not pass it and then stop.\n"
            "if there is a pedestian on the crosswalk or jaywalking, the ego car should stop right away and only move after they have completely crossed the street.\n"
            "\"i\" in the explanations refers to the ego car.\n"
            f"Question: {question}\n"
            f"Answer: {qa['answer']}\n"
            f"Explanation: \"{explanation}\"?\n"
            "make your answer as concise as possible, and do not repeat the question, answer and explanation.\n"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_uri,
                        "max_pixels": 360 * 420,
                        "fps": 10.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Format input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        ).to(model.device)

        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=350,  do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        print(f"\n📼 Video: {fname}")
        print(f"❓ Question: {question}")
        print(f"answer: {qa['answer']}")
        print(f"💬 explanation: {explanation}")
        print(f"🟢 Output: {output}")

        all_outputs.append({
            "video": fname,
            "question": question,
            "prompt": prompt,
            "output": output
        })

    
with open(output_json_path, "w") as f:
    json.dump(all_outputs, f, indent=4)


# # Load the outputs from the JSON file
# with open("outputs.json", "r") as f:
#     all_outputs = json.load(f)

scores = []
for item in all_outputs:
    output_text = item["output"]
    # Extract score: search for "score:" followed by optional whitespace and a digit (0 or 1)
    # match = re.search(r"Score:\s*([01])", output_text)
    match = re.search(r"Final Score:\s*(?:\*\*)?([01])(?:\*\*)?", output_text)
    if match:
        score = int(match.group(1))
        scores.append(score)
    else:
        print("Warning: score not found in output:", output_text)

# Calculate the total score and number of questions
total_questions = len(scores)
total_score = sum(scores)
print("scores:", scores)
print(f"Total score: {total_score} out of {total_questions} questions.")