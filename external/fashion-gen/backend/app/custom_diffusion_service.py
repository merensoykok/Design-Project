import os
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from PIL import Image
import base64
from io import BytesIO
import logging

# Set cache directory and disable telemetry for faster loading
cache_dir = os.path.expanduser("~/.cache/huggingface")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online but prefer cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomControlNetDiffusionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize pipeline as None - it'll be loaded when needed
        self.controlnet_pipeline = None
        
        # Model configurations from your notebook
        self.controlnet_model_id = "menevseyup/cnet-inpaint-15-05-2025"
        self.base_model_id = "botp/stable-diffusion-v1-5-inpainting"
    
    def load_controlnet_pipeline(self):
        """Load the ControlNet inpainting pipeline"""
        if self.controlnet_pipeline is None:
            logger.info("Loading Custom ControlNet inpainting pipeline...")
            
            # Check if models are already cached to avoid re-downloading
            controlnet_cache_path = os.path.join(cache_dir, f"models--{self.controlnet_model_id.replace('/', '--')}")
            base_model_cache_path = os.path.join(cache_dir, f"models--{self.base_model_id.replace('/', '--')}")
            
            if os.path.exists(controlnet_cache_path):
                logger.info(f"Found cached ControlNet model at {controlnet_cache_path}")
            else:
                logger.info(f"ControlNet model not cached, will download to {controlnet_cache_path}")
                
            if os.path.exists(base_model_cache_path):
                logger.info(f"Found cached base model at {base_model_cache_path}")
            else:
                logger.info(f"Base model not cached, will download to {base_model_cache_path}")
            
            try:
                # Load ControlNet model with better caching
                logger.info(f"Loading ControlNet from {self.controlnet_model_id}")
                controlnet = ControlNetModel.from_pretrained(
                    self.controlnet_model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    cache_dir=cache_dir,
                    local_files_only=False,  # Try local first, download if needed
                    force_download=False,
                    resume_download=True
                )
                
                # Load the ControlNet Inpainting Pipeline with better caching
                logger.info(f"Loading base inpainting model from {self.base_model_id}")
                self.controlnet_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    self.base_model_id,
                    controlnet=controlnet,
                    safety_checker=None,
                    requires_safety_checker=False,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    cache_dir=cache_dir,
                    local_files_only=False,  # Try local first, download if needed
                    force_download=False,
                    resume_download=True
                ).to(self.device)
                
                # Use UniPCMultistepScheduler as in your notebook
                self.controlnet_pipeline.scheduler = UniPCMultistepScheduler.from_config(
                    self.controlnet_pipeline.scheduler.config
                )
                
                # Enable memory efficient features if using CUDA
                if self.device == "cuda":
                    try:
                        self.controlnet_pipeline.enable_attention_slicing()
                        self.controlnet_pipeline.enable_model_cpu_offload()
                    except Exception as e:
                        logger.warning(f"Could not enable memory optimizations: {e}")
                    
                logger.info("Custom ControlNet inpainting pipeline loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading ControlNet inpainting pipeline: {e}")
                raise
    
    def generate_controlnet_inpaint_image(self, original_image_data, mask_data, prompt, negative_prompt=None, num_inference_steps=30, guidance_scale=7.5):
        """Generate an inpainted fashion image using your custom ControlNet model"""
        try:
            # Load pipeline if not already loaded
            self.load_controlnet_pipeline()
            
            # Debug logging
            logger.info(f"Received original_image_data type: {type(original_image_data)}")
            logger.info(f"Received mask_data type: {type(mask_data)}")
            
            # Validate inputs
            if original_image_data is None:
                raise ValueError("original_image_data cannot be None")
            if mask_data is None:
                raise ValueError("mask_data cannot be None")
            
            # Process original image
            if original_image_data.startswith('data:image'):
                original_image_data = original_image_data.split(',')[1]
            original_image = Image.open(BytesIO(base64.b64decode(original_image_data))).convert("RGB")
            logger.info(f"Original image size: {original_image.size}")
            
            # Process mask - this will be used as both mask and controlnet conditioning
            if mask_data.startswith('data:image'):
                mask_data = mask_data.split(',')[1]
            mask_image = Image.open(BytesIO(base64.b64decode(mask_data))).convert("L")  # Convert to grayscale
            logger.info(f"Mask image size: {mask_image.size}")
            
            # Resize images to 512x512 as in your notebook
            original_image = original_image.resize((512, 512), Image.LANCZOS)
            mask_image = mask_image.resize((512, 512), Image.LANCZOS)
            logger.info(f"Resized images to 512x512")
            
            # Validate processed images
            if original_image is None:
                raise ValueError("Processed original_image is None")
            if mask_image is None:
                raise ValueError("Processed mask_image is None")
            
            # Debug the processed images
            logger.info(f"Original image after processing - Type: {type(original_image)}, Size: {original_image.size if hasattr(original_image, 'size') else 'N/A'}")
            logger.info(f"Mask image after processing - Type: {type(mask_image)}, Size: {mask_image.size if hasattr(mask_image, 'size') else 'N/A'}")
            
            # Default negative prompt for better fashion results
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, watermark, signature, text, poor fabric texture, unrealistic clothing"
            
            # Enhanced prompt for better fashion results
            enhanced_prompt = f"high quality, detailed, professional fashion, {prompt}, realistic fabric texture, well-fitted clothing, natural lighting"
            
            logger.info(f"Generating ControlNet inpainted image with prompt: {enhanced_prompt}")
            
            # Generate image with proper context manager (based on your notebook logic)
            with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
                # Set generator for reproducible results
                generator = torch.Generator(device=self.device).manual_seed(42)
                
                # Use the correct parameter names from your notebook
                result = self.controlnet_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    image=original_image,
                    mask_image=mask_image,
                    control_image=mask_image,  # FIXED: Use control_image instead of controlnet_conditioning_image
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=1.0,  # Add this parameter from your notebook
                    generator=generator
                )
            
            image = result.images[0]
            
            # Convert to base64 for frontend
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info("Custom ControlNet inpainted image generated successfully!")
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error generating custom ControlNet inpainted image: {e}")
            raise

# Global instance
custom_diffusion_service = CustomControlNetDiffusionService() 