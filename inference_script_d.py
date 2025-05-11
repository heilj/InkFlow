import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchdiffeq import odeint
import os

# Import necessary modules
from diffusers import AutoencoderKL
from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from networks.unet_cfg import UNetGenerator
from networks.utils import letter2index, con_symbols


def optimized_scale(positive, negative):
    """
    Compute optimized scaling factor for classifier-free guidance.
    positive: v_cond, shape (B, *)
    negative: v_uncond, shape (B, *)
    Returns: s_star, shape (B, 1)
    """
    dot = torch.sum(positive * negative, dim=1, keepdim=True)        # (B, 1)
    norm_sq = torch.sum(negative ** 2, dim=1, keepdim=True) + 1e-8   # Avoid div 0
    return dot / norm_sq  # (B, 1)


class WrappedModelCFG(ModelWrapper):
    """Wrapped model for ODE solver with CFG support"""
    def __init__(self, model, cfg_scale_text=2.5, cfg_scale_style=1.5):
        super().__init__(model)
        self.cfg_scale_text = cfg_scale_text
        self.cfg_scale_style = cfg_scale_style

    def forward(self, x, t, **extras):
        # ODE solver gives scalar t, but UNet expects shape [B, 1]
        if t.dim() == 0:
            t = t.expand(x.size(0)).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        content = extras["content"]
        style = extras["style"]
        
        # Get conditional and unconditional predictions
        v_cond, v_uncond, v_null = self.model(
            x_t=x, 
            content=content, 
            style=style,
            t=t, 
            tag='sample',
            cfg_scale=1.5
        )

        # Flatten tensors for optimized scaling
        v_cond_flat = v_cond.view(v_cond.size(0), -1)
        v_uncond_flat = v_uncond.view(v_uncond.size(0), -1)
        v_null_flat = v_null.view(v_null.size(0), -1)
        
        # Compute optimized scales
        s_star_base = optimized_scale(v_cond_flat, v_null_flat)
        s_star_style = optimized_scale(v_uncond_flat, v_null_flat)

        # Reshape for broadcasting
        while s_star_base.dim() < v_uncond.dim():
            s_star_base = s_star_base.unsqueeze(-1)
        while s_star_style.dim() < v_uncond.dim():
            s_star_style = s_star_style.unsqueeze(-1)

        # Apply CFG-Zero*
        if t[0].item() < 1e-5:  # Check if we're at the initial time step
            v_guided = torch.zeros_like(v_uncond)  # For initial step
        else:
            # Guided velocity with optimized scale
            v_guided = v_null * (s_star_base + s_star_style)/2 +\
                self.cfg_scale_text * (v_uncond - v_null * s_star_base) + \
                self.cfg_scale_style * (v_cond - v_null * s_star_style)
            # v_guided = v_null + self.cfg_scale_text * (v_uncond -v_null) + self.cfg_scale_style * (v_cond - v_null)
                
        return v_guided


def preprocess_style_image(img_path, device, target_height=64):
    """Preprocess a style reference image from file"""
    # Load image
    if isinstance(img_path, str):
        img = Image.open(img_path).convert('L')  # Convert to grayscale
    else:
        img = img_path
        
    # Resize maintaining aspect ratio
    w, h = img.size
    new_w = int(w * (target_height / h))
    img = img.resize((new_w, target_height), Image.LANCZOS)
    
    # Convert to tensor and normalize to [-1, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = (img_tensor * 2) - 1
    
    # Add batch and channel dimensions
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    return img_tensor


def prepare_content_tensor(text, device, char_width=32):
    """Prepare content tensor from text"""
    content = [letter2index.get(c, letter2index[' ']) for c in text]
    content_img = con_symbols[content]
    return content_img.unsqueeze(0).to(device)


def sample(
    unet,
    vae,
    content_text,
    style_image,
    device,
    output_name=None,
    output_folder=None,
    cfg_scale_text=2.5,
    cfg_scale_style=1.5,
    char_width=32,
    num_steps=100,
    step_size=0.02,
    ode_method="dopri5",
    show_intermediates=False,
    num_intermediates=5,
):
    """
    Generate a handwritten text sample in the style of the reference image.
    
    Args:
        unet: The UNet model
        vae: VAE model for decoding latents
        content_text: Text to generate
        style_image_path: Path to style reference image
        device: Computation device
        output_path: Path to save output image
        cfg_scale_text: Text guidance scale
        cfg_scale_style: Style guidance scale
        char_width: Character width in pixels
        num_steps: Number of ODE solver steps
        step_size: ODE solver step size
        ode_method: ODE solver method
        show_intermediates: Whether to show intermediate steps
        num_intermediates: Number of intermediate steps to show
    
    Returns:
        Generated image as PIL Image
    """
    # Prepare style reference

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder,output_name)
    if isinstance(style_image, str):
        style_img = preprocess_style_image(style_image, device)
    
    # Create a 2-sample style batch 
    # (model expects 2 style references but we'll use the same one twice)
    max_w = style_img.shape[-1]
    if style_image.shape[0] == 2:
        style_images = torch.ones(2, style_img.shape[-2], max_w, dtype=torch.float32, device=device)
        style_images[0] = style_img[0, 0]
        style_images[1] = style_img[0, 0]  # duplicate the same style
    style_image = style_image.to(device)
    style_images = style_images.unsqueeze(0)
    
    # Prepare content (text)
    content_img = prepare_content_tensor(content_text, device, char_width)
    
    # Calculate latent size based on text length
    text_len = len(content_text)
    latent_width = text_len * char_width // 8  # VAE downsamples by factor of 8
    
    # Initialize starting noise
    x0 = torch.randn(1, 4, 64//8, latent_width, device=device)
    
    # Wrap model with CFG guidance
    flow = unet
    wrapped_model = WrappedModelCFG(
        flow, 
        cfg_scale_text=cfg_scale_text,
        cfg_scale_style=cfg_scale_style
    )
    
    # Setup ODE solver
    time_steps = torch.linspace(0, 1, num_steps).to(device)
    solver = ODESolver(wrapped_model)
    
    # Sample with ODE solver
    print(f"Generating '{content_text}' with text CFG={cfg_scale_text}, style CFG={cfg_scale_style}")
    sol = solver.sample(
        x_init=x0,
        step_size=step_size,
        method=ode_method,
        time_grid=time_steps,
        return_intermediates=show_intermediates,
        content=content_img,
        style=style_images,
    )
    
    # Convert to images
    if show_intermediates:
        # Get selected intermediate steps
        indices = np.linspace(0, len(sol)-1, num_intermediates, dtype=int)
        intermediate_images = []
        
        for idx in indices:
            latents = 1 / 0.18215 * sol[idx].to(device)
            img = vae.decode(latents).sample
            intermediate_images.append(img[0, 0].cpu().detach())
            
        # Create grid of intermediates
        fig, axes = plt.subplots(1, num_intermediates, figsize=(num_intermediates*2, 2))
        for i, ax in enumerate(axes):
            ax.imshow(intermediate_images[i], cmap='gray')
            ax.set_title(f"t={i/(num_intermediates-1):.2f}")
            ax.axis('off')
        plt.tight_layout()
        
        if output_path:
            intermediates_path = str(output_path).replace('.png', '_intermediates.png')
            plt.savefig(intermediates_path)
            print(f"Intermediate steps saved to {intermediates_path}")
        else:
            plt.show()
    
    # Get final image
    final_img = sol[-1].detach() if show_intermediates else sol.detach()
    latents = 1 / 0.18215 * final_img
    img = vae.decode(latents.to(device)).sample
    final_image = img[0, 0].cpu().detach()
    
    # Convert to PIL image
    final_pil = Image.fromarray(
        ((final_image.numpy() + 1) / 2 * 255).astype(np.uint8)
    )
    
    # Save output if path provided
    if output_path:
        final_pil.save(output_path)
        print(f"Output saved to {output_path}")
    
    return final_pil


def main():
    parser = argparse.ArgumentParser(description="Handwriting Style Transfer Inference")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--style_image", type=str, required=True, help="Path to style reference image")
    parser.add_argument("--text", type=str, required=True, help="Text to generate")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--output_dir", type=str, default=".results/", help="Output image folder")
    
    parser.add_argument("--text_cfg", type=float, default=2.5, help="Text guidance scale")
    parser.add_argument("--style_cfg", type=float, default=1.5, help="Style guidance scale")
    parser.add_argument("--char_width", type=int, default=32, help="Character width in pixels")
    
    parser.add_argument("--steps", type=int, default=200, help="Number of sampling steps")
    parser.add_argument("--step_size", type=float, default=0.02, help="ODE solver step size")
    parser.add_argument("--method", type=str, default="dopri5", help="ODE solver method")
    
    parser.add_argument("--show_intermediates", action="store_true", help="Show intermediate steps")
    parser.add_argument("--num_intermediates", type=int, default=5, help="Number of intermediate steps to show")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained('stable-diffusion-v1-5', subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)
    
    # Load UNet model
    print(f"Loading model from {args.model_path}...")
    unet = UNetGenerator(input_channels=4, base_channels=512, context_dim=512).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Try different keys that might exist in the checkpoint
    if 'unet_ema' in checkpoint:
        print("Loading from EMA weights")
        # Need to extract state dict from EMA format
        if 'shadow' in checkpoint['unet_ema']:
            unet.load_state_dict(checkpoint['unet_ema']['shadow'])
        else:
            # Directly load EMA weights
            unet.load_state_dict(checkpoint['unet_ema'])
    elif 'unet' in checkpoint:
        print("Loading from regular weights")
        unet.load_state_dict(checkpoint['unet'])
    else:
        print("Trying to load weights directly")
        unet.load_state_dict(checkpoint)
    
    unet.eval()
    
    # Generate image
    with torch.no_grad():
        sample(
            unet=unet,
            vae=vae,
            content_text=args.text,
            style_image_path=args.style_image,
            device=device,
            output_path=args.output,
            cfg_scale_text=args.text_cfg,
            cfg_scale_style=args.style_cfg,
            char_width=args.char_width,
            num_steps=args.steps,
            step_size=args.step_size,
            ode_method=args.method,
            show_intermediates=args.show_intermediates,
            num_intermediates=args.num_intermediates,
        )
    
    print("Done!")


if __name__ == "__main__":
    main()