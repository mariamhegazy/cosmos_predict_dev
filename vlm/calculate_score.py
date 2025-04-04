import json 
import re

json_file_path = "/capstor/store/cscs/swissai/a03/mariam/DrivingVQA/outputs.json"
# Load the outputs from the JSON file
with open(json_file_path, "r") as f:
    all_outputs = json.load(f)

scores = []
for item in all_outputs:
    output_text = item["output"]
    # Extract score: search for "score:" followed by optional whitespace and a digit (0 or 1)
    # match = re.search(r"Final Score:\s*([01])", output_text)
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