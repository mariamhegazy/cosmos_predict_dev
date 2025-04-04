import json
import re
import os
from sklearn.metrics import cohen_kappa_score

human_rating_json_path = "/capstor/store/cscs/swissai/a03/mariam/DrivingVQA/splits/transformed_test_data _filtered_human.json"
vlm_rating_json_path = "/capstor/store/cscs/swissai/a03/mariam/DrivingVQA/outputs_21.json"

with open(human_rating_json_path, "r") as f:
    human_data = json.load(f)

with open(vlm_rating_json_path, "r") as f:
    vlm_data = json.load(f)


human_scores = {}
for key, entries in human_data.items():
    for entry in entries:
        # Convert the score to float (or int) if needed
        try: 
            human_scores[entry["img_filename"]] = float(entry["human_score"])
        except:
            print(f"Error converting score for {entry['img_filename']}")
            continue

vlm_scores = {}
for entry in vlm_data:
    video_filename = entry.get("video", "")
    # Example: "DrivingTest_0146_000013.mp4"

    parts = os.path.splitext(video_filename)[0].split("_")
    if len(parts) >= 2:
        img_num = parts[1]
        img_filename = f"{img_num}.jpg"
    else:
        print(f"Skipping video with unexpected format: {video_filename}")
        continue

    output_text = entry.get("output", "")
    match = re.search(r"Final Score:\s*(?:\*\*)?([\d\.]+)(?:\*\*)?", output_text)
    if match:
        vlm_score = float(match.group(1))
    else:
        print(f"No score found for video: {video_filename}")
        continue

    vlm_scores[img_filename] = vlm_score


common_filenames = set(human_scores.keys()).intersection(vlm_scores.keys())
if not common_filenames:
    print("No common image filenames found between the two JSON files.")
    exit()

human_list = []
vlm_list = []
for img in common_filenames:
    human_list.append(human_scores[img])
    vlm_list.append(vlm_scores[img])

print("Common filenames:", common_filenames)
print("Human scores:", human_list)
print("VLM scores:", vlm_list)

mapping = {0.0: 0, 0.5: 1, 1.0: 2}

# Convert the human and VLM scores to discrete labels
human_discrete = [mapping[score] for score in human_list]
vlm_discrete = [mapping[score] for score in vlm_list]

# Compute Cohen's Kappa using the discrete labels
kappa = cohen_kappa_score(human_discrete, vlm_discrete)
print("Cohen's Kappa:", kappa)
