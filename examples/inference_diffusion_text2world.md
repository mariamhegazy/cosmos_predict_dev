## Inference with diffusion-based Text2World models

### Environment setup

Clone the `cosmos-predict1` source code
```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict1.git
cd cosmos-predict1
```

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.10.x`. Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

```bash
# Create the cosmos-predict1 conda environment.
conda env create --file cosmos-predict1.yaml
# Activate the cosmos-predict1 conda environment.
conda activate cosmos-predict1
# Install the dependencies.
pip install -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
```

You can test the environment setup with
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B 14B --model_types Text2World
   ```

### Examples

There are two models available for diffusion world generation from text input: `Cosmos-Predict1-7B-Text2World` and `Cosmos-Predict1-14B-Text2World`.

The inference script is `cosmos_predict1/diffusion/inference/text2world.py`.
It requires the input argument `--prompt` (text input).
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py --help
```

We will set the prompt with an environment variable first.
```bash
PROMPT="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
```

#### Example 1: single generation
This is the basic example for running inference on the 7B model with a single prompt.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --prompt "${PROMPT}" \
    --offload_prompt_upsampler \
    --video_save_name diffusion-text2world-7b
```

#### Example 2: single generation on the 14B model with model offloading
We run inference on the 14B model with offloading flags enabled. This is suitable for low-memory GPUs. Model offloading is also required for the 14B model to avoid OOM.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-14B-Text2World \
    --prompt "${PROMPT}" \
    --offload_tokenizer \
    --offload_diffusion_transformer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-text2world-14b
```

#### Example 3: single generation with multi-GPU inference
This example runs parallelized inference on a single prompt using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/text2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --prompt "${PROMPT}" \
    --offload_prompt_upsampler \
    --video_save_name diffusion-text2world-7b-8gpu
```

#### Example 4: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
The JSONL file should contain one prompt per line in the following format, where each line must contain a `prompt` field:
```json
{"prompt": "prompt1"}
{"prompt": "prompt2"}
```
Inference command:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World \
    --batch_input_path assets/diffusion/batch_inputs/text2world.jsonl \
    --offload_prompt_upsampler \
    --video_save_folder diffusion-text2world-7b-batch
```
