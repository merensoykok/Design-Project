from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import AzureOpenAI
import os
import base64
from dotenv import load_dotenv
from typing import Optional
import asyncio
from .diffusion_service import diffusion_service
from .custom_diffusion_service import custom_diffusion_service

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bitirme-odevi.openai.azure.com/")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

system_prompt = """You are an AI stylist and fashion prompt engineer. Your job is to translate a casual or vague user request into a detailed, specific outfit description suitable for a Stable Diffusion model.

Guidelines:
- Focus only one part of the outfit and clothing attributes (that user wants to change). Ignore background, lighting, photo style, camera settings.
- Use the user's message to infer context (weather, event, intent) and adjust the outfit accordingly.
- Your output should describe:
  - Garment type(s) (e.g., T-shirt, formal jacket, linen shirt)
  - Gender and body type (if inferable)
  - Fit (tight, relaxed, oversized)
  - Sleeve length, neckline, tucking
  - Length (cropped, hip, long)
  - Material (e.g., cotton, denim, wool)
  - Color(s), pattern (e.g., striped, floral)
  - Formality (casual, formal, semi-formal)
- Write the prompt as a comma-separated string, describing what the person is wearing
- Do not describe face, hair, background, mood, or artistic style

Respond with only the prompt — no explanations, notes, or extra words."""

class PromptRequest(BaseModel):
    prompt: Optional[str] = None
    imageData: Optional[str] = None
    maskData: Optional[str] = None
    originalImage: Optional[str] = None

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Fashion Generation API with Custom ControlNet"}

@app.get("/test-diffusion")
async def test_diffusion():
    """Test endpoint to verify Stable Diffusion is working"""
    try:
        test_prompt = "red cotton t-shirt, casual fit, short sleeves"
        print(f"Testing Stable Diffusion with prompt: {test_prompt}")
        
        generated_image = await asyncio.get_event_loop().run_in_executor(
            None, 
            diffusion_service.generate_fashion_image,
            test_prompt
        )
        
        return {
            "status": "success",
            "message": "Stable Diffusion is working correctly!",
            "testImage": generated_image
        }
    except Exception as e:
        print(f"Stable Diffusion test failed: {str(e)}")
        return {
            "status": "error", 
            "message": f"Stable Diffusion test failed: {str(e)}"
        }

@app.get("/test-custom-controlnet")
async def test_custom_controlnet():
    """Test endpoint to verify Custom ControlNet is working"""
    try:
        # This would need actual image data to test properly
        return {
            "status": "info",
            "message": "Custom ControlNet service is loaded. Use /generate with mask data to test inpainting."
        }
    except Exception as e:
        print(f"Custom ControlNet test failed: {str(e)}")
        return {
            "status": "error", 
            "message": f"Custom ControlNet test failed: {str(e)}"
        }

@app.post("/generate")
async def generate_fashion(request: Request, body: PromptRequest):
    user_input = body.prompt
    user_image = body.imageData
    mask_data = body.maskData
    original_image = body.originalImage
    
    # Handle mask data case - use CUSTOM ControlNet model for inpainting
    if mask_data and original_image:
        try:
            # First get the GPT-generated fashion prompt
            if user_input:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=200,
                    temperature=0.6,
                    top_p=1.0,
                    model=deployment
                )
                generated_prompt = response.choices[0].message.content.strip()
                print(f"Generated Prompt for Custom ControlNet: {generated_prompt}")
                print(f"Starting Custom ControlNet inpainting...")
                
                # Generate inpainted image using YOUR CUSTOM ControlNet model
                generated_image = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    custom_diffusion_service.generate_controlnet_inpaint_image,
                    original_image,  # This is original_image_data parameter
                    mask_data,
                    generated_prompt
                )
                
                print(f"Custom ControlNet inpainting completed!")
                return {
                    "response": generated_prompt, 
                    "generatedImage": generated_image,
                    "type": "custom_controlnet_inpaint"
                }
            else:
                # Just return the mask for testing if no prompt provided
                print(f"Received mask data for Custom ControlNet processing without text prompt")
                return {"response": "Mask received for Custom ControlNet processing", "maskUrl": mask_data}
        except Exception as e:
            print(f"Error processing mask with Custom ControlNet: {str(e)}")
            return {"response": f"Error processing mask with Custom ControlNet: {str(e)}", "maskUrl": mask_data}
    
    # Handle text-only case - generate image from scratch using standard SD
    if user_input and not user_image and not mask_data:
        try:
            # Get GPT-generated fashion prompt
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=200,
                temperature=0.6,
                top_p=1.0,
                model=deployment
            )
            generated_prompt = response.choices[0].message.content.strip()
            print(f"Generated Prompt: {generated_prompt}")
            print(f"Starting Stable Diffusion text-to-image generation...")
            
            # Generate image using standard Stable Diffusion
            generated_image = await asyncio.get_event_loop().run_in_executor(
                None, 
                diffusion_service.generate_fashion_image,
                generated_prompt
            )
            
            print(f"Stable Diffusion text-to-image completed!")
            return {
                "response": generated_prompt,
                "generatedImage": generated_image,
                "type": "text2img"
            }
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return {"response": f"Error generating image: {str(e)}", "imageUrl": None}
    
    # Legacy: For testing, if user sends both text and image, return output.jpeg
    if user_input and user_image and not mask_data:
        output_url = str(request.base_url) + "static/output.jpeg"
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=200,
                temperature=0.6,
                top_p=1.0,
                model=deployment
            )
            generated_prompt = response.choices[0].message.content.strip()
            print(f"Generated Prompt (legacy): {generated_prompt}")
            return {"response": generated_prompt, "imageUrl": output_url}
        except Exception as e:
            return {"response": f"Error generating prompt: {str(e)}", "imageUrl": output_url}
    
    # Handle image-only case
    if user_image and not user_input and not mask_data:
        return {"response": "Image received", "imageUrl": user_image}
    
    # If we get here, neither text nor image was provided
    return {"response": "Please provide either text or an image."} 