import os
import torch
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import StableDiffusionPipeline


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model_name = os.getenv("MODEL_NAME")
    model = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments

    endpoint = model_inputs.get('endpoint',None)
    params =  model_inputs.get('params', None)

    if params == None:
        return {'message': "No params provided"}


    
    prompt = params.get('prompt', None)
    negative_prompt = params.get('negative_prompt', None)
    steps = params.get('steps', 50)
    cfg_scale = params.get('cfg_scale', 7)
    batch_size = params.get('batch_size',1)
    width = params.get('width',384)
    height = params.get('height',384)
    n_iter = params.get('n_iter',1)
    seed = params.get('seed',-1)




   
    
    # Run the model
    result = model(prompt, 
                   steps=steps, 
                   scale=cfg_scale, 
                   guidance_scale=cfg_scale,
                   batch_size=batch_size,
                   negative_prompt=negative_prompt,
                   width=width,
                   height=height,
                   num_inference_steps=n_iter,
                   prng_seed=seed,
                      )

    # Check if result is an image or text
    image = result.images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'images': [image_base64]}


# "endpoint": "txt2img",
#   "params": {
#     "prompt": "home",
#     "negative_prompt": "low quality",
#     "steps": 20,
#     "sampler_name": "Euler a",
#     "cfg_scale": 7.5,
#     "seed": -1,
#     "batch_size": 1,
#     "n_iter": 1,
#     "width": 768,
#     "height": 768,
#     "tiling": False
#   }