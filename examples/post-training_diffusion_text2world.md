## Post-training diffusion-based Text2World models

### Model Support Matrix

We support the following Cosmos Diffusion models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-7B-Text2World           | **Supported**    | 8 NVIDIA GPUs*                           |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

### Environment setup

Please refer to the Post-training section of [INSTALL.md](/INSTALL.md#post-training) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Text2World
   ```

### Examples

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your use case.

There are 3 steps to post-training: downloading a dataset, preprocessing the data, and post-training the model.

#### 1. Download a Dataset

The first step is to download a dataset with videos and captions.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

Example 1. You can use [nvidia/Cosmos-NeMo-Assets](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) for post-training.

```bash
mkdir -p datasets/cosmos_nemo_assets/

# This command will download the videos for physical AI
huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir datasets/cosmos_nemo_assets/ --include "*.mp4*"

mv datasets/cosmos_nemo_assets/nemo_diffusion_example_data datasets/cosmos_nemo_assets/videos
```

#### 2. Preprocessing the Data

Run the following command to pre-compute T5-XXL embeddings for the video captions used for post-training:

```bash
# The script will use the provided prompt, save the T5-XXL embeddings in pickle format.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings_from_cosmos_nemo_assets.py --dataset_path datasets/cosmos_nemo_assets --prompt "A video of sks teal robot."
```

Dataset folder format:
```
datasets/cosmos_nemo_assets/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
```

#### 3. Post-train the Model

Run the following command to execute an example post-training job with the above data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train --config=cosmos_predict1/diffusion/training/config/config.py -- experiment=text2world_7b_example_cosmos_nemo_assets
```

The model will be post-trained using the above cosmos_nemo_assets dataset.
See the config `text2world_7b_example_cosmos_nemo_assets` defined in `cosmos_predict1/diffusion/training/config/text2world/experiment.py` to understand how the dataloader is determined.
```python
num_frames = 121
example_video_dataset_cosmos_nemo_assets = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    sequence_interval=1,
    num_frames=num_frames,
    video_size=(720, 1280),
    start_frame_interval=1,
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
)
...

text2world_7b_example_cosmos_nemo_assets = LazyDict(
    dict(
        ...
        dataloader_train=dataloader_train_cosmos_nemo_assets,
        ...
    )
)
...

```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_text2world`, `NAME` is `text2world_7b_example_cosmos_nemo_assets`.

See the job config to understand how they are determined.
```python
text2world_7b_example_cosmos_nemo_assets = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_7b_example_cosmos_nemo_assets",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_text2world/text2world_7b_example_cosmos_nemo_assets/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```


### Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_diffusion_text2world.md](/examples/inference_diffusion_text2world.md).

#### 1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Text2World_post-trained/model.pt`

For example, if a post-trained checkpoint (ema) with 1000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Text2World_post-trained/
cp checkpoints/posttraining/diffusion_text2world/text2world_7b_example_cosmos_nemo_assets/checkpoints/iter_000001000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Text2World_post-trained/model.pt
```
#### 2. Running the Inference

We will set the prompt with an environment variable first.
```bash
PROMPT="A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. \
The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. \
A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, \
suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. \
The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of \
field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."
```

```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/text2world.py --diffusion_transformer_dir Cosmos-Predict1-7B-Text2World_post-trained --prompt "${PROMPT}" --video_save_name diffusion-text2world-7b-post-train --offload_prompt_upsampler
```
