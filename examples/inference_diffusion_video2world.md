## Inference with diffusion-based Video2World models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B 14B --model_types Video2World
   ```

### Examples

There are two models available for diffusion world generation from text and image/video input: `Cosmos-Predict1-7B-Video2World` and `Cosmos-Predict1-14B-Video2World`.

The inference script is `cosmos_predict1/diffusion/inference/video2world.py`.
It requires the input argument `--input_image_or_video_path` (image/video input); if the prompt upsampler is disabled, `--prompt` (text input) must also be provided.
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py --help
```

#### Example 1: single generation
This is the basic example for running inference on the 7B model with a single image. No text prompts are provided here.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b
```

#### Example 2: single generation on the 14B model with model offloading
We run inference on the 14B model with offloading flags enabled. This is suitable for low-memory GPUs. Model offloading is also required for the 14B model to avoid OOM.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-14B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_tokenizer \
    --offload_diffusion_transformer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-video2world-14b
```

#### Example 3: single generation with multi-GPU inference
This example runs parallelized inference on a single prompt using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/video2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b
```

#### Example 4: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
Each line in the JSONL file must contain a `visual_input` field:
```json
{"visual_input": "path/to/video1.mp4"}
{"visual_input": "path/to/video2.mp4"}
```
Inference command (with 9 input frames):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --batch_input_path assets/diffusion/batch_inputs/video2world_ps.jsonl \
    --num_input_frames 9 \
    --offload_prompt_upsampler \
    --video_save_folder diffusion-video2world-7b-batch
```

#### Example 5: batch generation without prompt upsampler
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
The prompt upsampler is disabled, and thus each line in the JSONL file will need to include both `prompt` and `visual_input` fields.
```json
{"prompt": "prompt1", "visual_input": "path/to/video1.mp4"}
{"prompt": "prompt2", "visual_input": "path/to/video2.mp4"}
```
Inference command (with 9 input frames):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --batch_input_path assets/diffusion/batch_inputs/video2world_wo_ps.jsonl \
    --num_input_frames 9 \
    --disable_prompt_upsampler \
    --video_save_folder diffusion-video2world-7b-batch-wo-ps
```
