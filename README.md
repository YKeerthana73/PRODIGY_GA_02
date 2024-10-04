# PRODIGY_GA_02
# Stable Diffusion Image Generation from Text Prompts

This repository demonstrates how to use the Stable Diffusion model to generate images based on textual prompts. The notebook uses the `diffusers` library along with `transformers`, `gradio`, and `accelerate` to create and visualize images.

## Requirements

Make sure you have the following packages installed:


## Installation

```bash
pip install diffusers transformers gradio accelerate
```
## Importing the libraries
```bash
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch
```
##  Load the Model

Choose and load your preferred Stable Diffusion model. In this example, we will use the dreamlike-art model
``` bash
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

# Load the pipeline with the desired model
pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)

# Move the pipeline to the GPU for faster processing
pipe = pipe.to("cuda")
```

## Generate Images from Prompts

You can generate images by defining a text prompt. Below are examples of generating images based on two different prompts.

### Example 1: Generating an Image of a Rolls Royce
``` bash
# Define your first prompt
prompt = """dreamlikeart, a black rolls royce Ghost standing near foggy mountains in rain"""

# Generate the image
image = pipe(prompt).images[0]

# Display the generated image
print("[PROMPT]: ", prompt)
plt.imshow(image)
plt.axis('off')  # Hide the axes
plt.show()  # Show the image
```
### Example 2: Generating an Image of a Beach Vacation
``` bash
# Define your second prompt
prompt2 = """dreamlike, a group of people enjoying vacation at the beach"""

# Generate the image
image = pipe(prompt2).images[0]

# Display the generated image
print("[PROMPT]: ", prompt2)
plt.imshow(image)
plt.axis('off')  # Hide the axes
plt.show()  # Show the image
```


## Usage
```bash
This library is primarily used for generating images using diffusion models. 
It provides various pipelines for tasks like text-to-image generation.
```
## Note 
### Adjust the Collab Notebook's settings if you have a intel graphics card to ensure code runs smoothly.

```bash 
STEP 1 : GO TO RUNTIME 

STEP 2 : CLICK ON CHANGE RUNTIME TYPE

STEP 3 : AFTER THAT, CLICK ON THE T4 GPU AND SELECT SAVE TO SAVE THIS SETTING.

```
## Note 
### If you have NVIDIA GRAPHICS IN YOUR SYSTEM DO NOT CHANGE THE ABOVE MENTIONED CHANGES TO YOUR COLLAB NOTEBOOK
## The Google Colab Notebook offers a GPU to provide better graphics for text transformation into images, therefore please complete this work there.


