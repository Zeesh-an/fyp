from fastapi import APIRouter, Query, UploadFile, File, HTTPException, Body, Response
from fastapi.responses import FileResponse
from Promptist.inferencePromptist import Promptist
from Promptist.inferenceStableDiff import StableDiffusion
from PIL import Image, ImageDraw, ImageFont
import io
import os
import shutil
import uuid
import numpy as np
from PIL import Image

#Router for accessing promptist
class PromptistRouter(APIRouter):
    def __init__(self):
        super().__init__();
        self.router:APIRouter = APIRouter()
        self.router.add_api_route("/optimize-prompt-stable-diffusion/", self.optimize_prompt, methods = ["POST"])  
        self.promptist = Promptist()
    
    def optimize_prompt(self, inputData = Body()):
        # Here you would perform your optimization logic
        print(inputData)
        prompt = inputData["prompt"]
        optimized_prompt = self.promptist.generate(prompt)  # For demonstration, just converting to uppercase
        return {"optimized_prompt": optimized_prompt}


#Router for accessing Stable Diffusion
class StableDiffusionRouter(APIRouter):
    def __init__(self):
        super().__init__();
        self.router:APIRouter = APIRouter()
        self.router.add_api_route("/generate-image/", self.generateImage, methods = ["POST"])  
        self.stableDiffusion = StableDiffusion()
    
    def generateImage(self, inputData = Body()):
        # Here you would perform your optimization logic
        # print(inputData)
        prompt = inputData["prompt"]
        # generated_image = self.stableDiffusion.generateImage(prompt)  # For demonstration, just converting to uppercase
        img = Image.open("result001.jpeg")
        img = np.array(img)
        image_bytes = img.tobytes()
        # np_img = np.array(img).tobytes()
        return {"image_bytes" : image_bytes}


#Router for Prompt Discovery
class PromptDiscoveryRouter(APIRouter):
    def __init__(self):
        super().__init__();
        self.router:APIRouter = APIRouter()
        self.router.add_api_route("/discover-prompt/", self.findPrompt, methods = ["POST"])  
        self.stableDiffusion = StableDiffusion()
    
    def findPrompt(self, inputData = Body()):
        prompt = inputData["image_bytes"]
        generated_image = self.stableDiffusion.generateImage(prompt)  # For demonstration, just converting to uppercase
        generated_image = np.array(generated_image)[:,:,::-1]
        print(generated_image)
        # buf = io.BytesIO()
        # generated_image.save(buf, format='IMAGE/PNG')
        # byte_im = buf.getvalue()
        # print("Image Bytes", byte_im)
        return {"image_bytes" : "Success"}
