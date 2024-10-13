# PRODIGY_GA_02
# Image Generation with Pre-trained Models

This notebook demonstrates how to utilize pre-trained generative models like DALL-E mini and Stable Diffusion to create images from text prompts.

## Requirements

Make sure to have the following libraries installed:

```python
!pip install diffusers --upgrade
!pip install invisible-watermark transformers accelerate safetensors
```
# Getting Started

## Import Libraries

Begin by importing the necessary libraries to work with the model:

```python
import torch
from diffusers import StableDiffusionPipeline
```
# Load the Pre-trained Model

Use the code below to load the Stable Diffusion model:

```python
from diffusers import DiffusionPipeline

# Load the Stable Diffusion model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")  # Ensure the model is moved to GPU
```
# Define a Text Prompt

Create a text prompt that describes the image you want to generate:

```python
prompt = "An astronaut riding a green horse"
```
# Generate the Image

Use the pipeline to generate an image based on the prompt:

```python
images = pipe(prompt=prompt).images[0]
```
# Display the Image

Finally, display the generated image:

```python
images.show()
```

## Usage
```bash
This library is primarily used for generating images using diffusion models. 
It provides various pipelines for tasks like text-to-image generation.
```
# Example

For instance, using the prompt `"An astronaut riding a green horse"` will generate a unique image that matches this description.

## Notes

- Ensure you have access to a GPU for optimal performance.
- Feel free to experiment with different prompts to see a variety of generated images.



