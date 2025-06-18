import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import base64
from io import BytesIO
import logging

# Set cache directory for models to avoid re-downloading
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffusionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize pipelines as None - they'll be loaded when needed
        self.txt2img_pipeline = None
        self.inpaint_pipeline = None
        
        # Model configurations
        self.model_id = "runwayml/stable-diffusion-v1-5"  # Popular fashion-capable model
        self.inpaint_model_id = "runwayml/stable-diffusion-inpainting"
    
    def load_txt2img_pipeline(self):
        """Load the text-to-image pipeline"""
        if self.txt2img_pipeline is None:
            logger.info("Loading Stable Diffusion text-to-image pipeline...")
            try:
                self.txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,  # Disable safety checker for fashion images
                    requires_safety_checker=False,
                    cache_dir=os.path.expanduser("~/.cache/huggingface"),
                    force_download=False  # Don't re-download if already cached
                )
                self.txt2img_pipeline = self.txt2img_pipeline.to(self.device)
                
                # Enable memory efficient attention if using CUDA
                if self.device == "cuda":
                    try:
                        self.txt2img_pipeline.enable_attention_slicing()
                        self.txt2img_pipeline.enable_model_cpu_offload()
                    except:
                        logger.warning("Could not enable memory optimizations")
                    
                logger.info("Text-to-image pipeline loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading text-to-image pipeline: {e}")
                raise
    
    def load_inpaint_pipeline(self):
        """Load the inpainting pipeline"""
        if self.inpaint_pipeline is None:
            logger.info("Loading Stable Diffusion inpainting pipeline...")
            try:
                self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.inpaint_model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    cache_dir=os.path.expanduser("~/.cache/huggingface"),
                    force_download=False  # Don't re-download if already cached
                )
                self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
                
                if self.device == "cuda":
                    try:
                        self.inpaint_pipeline.enable_attention_slicing()
                        self.inpaint_pipeline.enable_model_cpu_offload()
                    except:
                        logger.warning("Could not enable memory optimizations")
                    
                logger.info("Inpainting pipeline loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading inpainting pipeline: {e}")
                raise
    
    def generate_fashion_image(self, prompt, negative_prompt=None, width=512, height=512, num_inference_steps=20, guidance_scale=7.5):
        """Generate a fashion image from text prompt"""
        try:
            # Load pipeline if not already loaded
            self.load_txt2img_pipeline()
            
            # Default negative prompt for better fashion images
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, watermark, signature, text"
            
            # Add fashion-specific positive prompt enhancements
            enhanced_prompt = f"high quality, detailed, professional fashion photography, {prompt}, clean background, well-lit, sharp focus"
            
            logger.info(f"Generating image with prompt: {enhanced_prompt}")
            
            # Generate image with proper context manager
            if self.device == "cuda":
                with torch.autocast(self.device):
                    result = self.txt2img_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(42)  # Fixed seed for consistency
                    )
            else:
                result = self.txt2img_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            image = result.images[0]
            
            # Convert to base64 for frontend
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info("Image generated successfully!")
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error generating fashion image: {e}")
            raise
    
    def generate_inpaint_image(self, original_image_data, mask_data, prompt, negative_prompt=None, num_inference_steps=20, guidance_scale=7.5):
        """Generate an inpainted fashion image using mask"""
        try:
            # Load inpainting pipeline if not already loaded
            self.load_inpaint_pipeline()
            
            # Process original image
            if original_image_data.startswith('data:image'):
                original_image_data = original_image_data.split(',')[1]
            original_image = Image.open(BytesIO(base64.b64decode(original_image_data))).convert("RGB")
            
            # Process mask
            if mask_data.startswith('data:image'):
                mask_data = mask_data.split(',')[1]
            mask_image = Image.open(BytesIO(base64.b64decode(mask_data))).convert("L")  # Convert to grayscale
            
            # Resize images to model requirements (512x512)
            original_image = original_image.resize((512, 512))
            mask_image = mask_image.resize((512, 512))
            
            # Default negative prompt
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, watermark, signature, text"
            
            # Enhanced prompt for inpainting
            enhanced_prompt = f"high quality, detailed, professional fashion, {prompt}, seamless blend, natural lighting"
            
            logger.info(f"Generating inpainted image with prompt: {enhanced_prompt}")
            
            # Generate inpainted image
            if self.device == "cuda":
                with torch.autocast(self.device):
                    result = self.inpaint_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        image=original_image,
                        mask_image=mask_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
            else:
                result = self.inpaint_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    image=original_image,
                    mask_image=mask_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            image = result.images[0]
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            logger.info("Inpainted image generated successfully!")
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error generating inpainted image: {e}")
            raise

# Global instance
diffusion_service = DiffusionService() 