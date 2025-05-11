import torch
import torch.nn.functional as F
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
import h5py
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Import necessary modules
from diffusers import AutoencoderKL
from flow_matching.solver import ODESolver
from networks.unet_cfg import UNetGenerator
from networks.utils import letter2index, con_symbols

# Import dataset and utilities from the original training code
from lib.datasets import Hdf5Dataset
from lib.alphabet import strLabelConverter
from tqdm import tqdm
from networks.ema import EMA

# Import from the inference script
from inference_script import (
    sample, WrappedModelCFG, preprocess_style_image, prepare_content_tensor
)

def load_writer_dataset(dataset_path, split='trnvalset_words64_OrgSz.hdf5', min_samples=1):
    """
    Load the dataset and organize samples by writer ID.
    
    Args:
        dataset_path: Path to the dataset directory
        split: Dataset split filename
        min_samples: Minimum number of samples required per writer
        
    Returns:
        Dictionary mapping writer IDs to lists of their samples
    """
    # Initialize transforms similar to training
    transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
    
    # Load dataset
    dataset = Hdf5Dataset(
        root=dataset_path,
        split=split,
        transforms=transforms,
        alphabet_key='iam_word'
    )
    train_len = int(0.1 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    l_valid_loader = DataLoader(
                val_set,
                # batch_sampler=batch_sampler,
                batch_size=2,
                shuffle=False,
                num_workers=8,
                collate_fn=Hdf5Dataset.collect_fn,
                drop_last=True,
                persistent_workers=True
            )
    # Collect samples by writer
    writer_samples = {}
    for batch in tqdm(l_valid_loader, desc="Preloading style samples"):
            # Process each item in batch to handle potentially different writer IDs
                images = batch['org_imgs']
                # print(images.shape)
                for i in range(len(batch['wids'])):
                    wid = int(batch['wids'][i])
                    if wid not in writer_samples:
                        writer_samples[wid] = []
                    # print(batch['org_imgs'][i].shape)
                    writer_samples[wid].append(batch['org_imgs'][i].clone())

    
    # Access the HDF5 file directly
    # with h5py.File(os.path.join(dataset_path, split), 'r') as f:
    #     wids = f['wids'][:]
    #     org_imgs = f['org_imgs'][:]
        
    #     # Group samples by writer ID
    #     for i in range(len(wids)):
    #         wid = int(wids[i])
    #         if wid not in writer_samples:
    #             writer_samples[wid] = []
            
    #         # Convert image data to tensor
    #         img = torch.from_numpy(org_imgs[i]).unsqueeze(0).float() / 127.5 - 1.0
    #         writer_samples[wid].append(img)
    
    # Filter writers with insufficient samples
    valid_writers = {wid: samples for wid, samples in writer_samples.items() 
                    if len(samples) >= min_samples}
    
    print(f"Found {len(valid_writers)} writers with at least {min_samples} samples")
    
    return valid_writers

def select_random_writer(writer_samples, exclude_wids=None):
    """
    Randomly select a writer and return their samples.
    
    Args:
        writer_samples: Dictionary mapping writer IDs to lists of their samples
        exclude_wids: List of writer IDs to exclude
        
    Returns:
        Tuple of (writer_id, list of samples)
    """
    available_wids = list(writer_samples.keys())
    
    if exclude_wids:
        available_wids = [wid for wid in available_wids if wid not in exclude_wids]
    
    if not available_wids:
        raise ValueError("No available writers after exclusion")
    
    selected_wid = random.choice(available_wids)
    return selected_wid, writer_samples[selected_wid]

def display_writer_samples(writer_id, samples, num_samples=4, title=None):
    """
    Display sample images from a writer.
    
    Args:
        writer_id: ID of the writer
        samples: List of sample tensors
        num_samples: Number of samples to display
        title: Optional title for the figure
    """
    fig, axes = plt.subplots(1, min(num_samples, len(samples)), figsize=(12, 3))
    
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < len(samples):
            ax.imshow(samples[i].squeeze().cpu(), cmap='gray')
            ax.axis('off')
    
    plt.suptitle(title or f"Writer ID: {writer_id}")
    plt.tight_layout()
    # plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Random Writer Style Transfer")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset_path", type=str, default='./data/iam', help="Path to the dataset directory")
    parser.add_argument("--dataset_split", type=str, default="testset_words64_OrgSz.hdf5", help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="random_writer_outputs", help="Output directory")
    
    # Text options
    parser.add_argument("--text", type=str, required=True, help="Text to generate")
    parser.add_argument("--num_writers", type=int, default=5, help="Number of random writers to sample")
    
    # Style and guidance parameters
    parser.add_argument("--cfg", type=float, default=3, help="cfg guidance scale")
    # parser.add_argument("--text_cfg", type=float, default=2.5, help="Text guidance scale")
    # parser.add_argument("--style_cfg", type=float, default=1.5, help="Style guidance scale")
    parser.add_argument("--char_width", type=int, default=32, help="Character width in pixels")
    
    # Generation parameters
    parser.add_argument("--steps", type=int, default=101, help="Number of sampling steps")
    parser.add_argument("--step_size", type=float, default=0.025, help="ODE solver step size")
    parser.add_argument("--method", type=str, default="dopri5", help="ODE solver method")
    
    # Writer selection options
    parser.add_argument("--seed", type=int, default=1001, help="Random seed for writer selection")
    parser.add_argument("--specific_wid", type=int, default=None, help="Use a specific writer ID instead of random selection")
    parser.add_argument("--min_samples", type=int, default=1, help="Minimum samples required for a writer")
    
    # Visualization options
    parser.add_argument("--show_references", action="store_true", help="Show reference samples from each writer")
    parser.add_argument("--create_grid", action="store_true", help="Create a grid of all generated samples")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load writer dataset
    print(f"Loading dataset from {args.dataset_path}...")
    writer_samples = load_writer_dataset(
        args.dataset_path, 
        split=args.dataset_split,
        min_samples=args.min_samples
    )
    
    # Load models
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained('stable-diffusion-v1-5', subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device)
    vae.eval()
    
    print(f"Loading model from {args.model_path}...")
    unet = UNetGenerator(input_channels=4, base_channels=512, context_dim=512).to(device)

    unet_ema  = EMA(unet)
    checkpoint = torch.load(args.model_path, map_location=device)
    state = checkpoint['unet_ema']['shadow'] if 'shadow' in checkpoint['unet_ema'] \
        else checkpoint['unet_ema']
    clean_state = {k.replace('model.', '', 1): v for k, v in state.items()}
    missing, unexpected = unet.load_state_dict(clean_state, strict=False)
    # print("missing:", missing, "unexpected:", unexpected)
    
    # Load model weights
    # if 'unet_ema' in checkpoint:
    #     print("Loading from EMA weights")
    #     if 'shadow' in checkpoint['unet_ema']:
    #         unet_ema.load_state_dict(checkpoint['unet_ema']['shadow'])
    #     else:
    #         unet_ema.load_state_dict(checkpoint['unet_ema'])
    # elif 'unet' in checkpoint:
    #     print("Loading from regular weights")
    #     unet.load_state_dict(checkpoint['unet'])
    # else:
    #     print("Trying to load weights directly")
    #     unet.load_state_dict(checkpoint)
    # unet_ema.copy_to_model()   
    unet.eval()
    
    # Process specific writer or random writers
    writer_ids = []
    if args.specific_wid is not None:
        if args.specific_wid in writer_samples:
            writer_ids = [args.specific_wid]
        else:
            print(f"Writer ID {args.specific_wid} not found or has insufficient samples")
            return
    else:
        # Select random writers without replacement
        available_wids = list(writer_samples.keys())
        num_writers = min(args.num_writers, len(available_wids))
        writer_ids = random.sample(available_wids, num_writers)
    
    # Generate samples for each writer
    generated_images = []
    writer_labels = []
    
    with torch.no_grad():
        for wid in writer_ids:
            writer_samples_list = writer_samples[wid]
            
            # Use first two samples as reference
            style_ref1 = writer_samples_list[0]
            # print(style_ref1.shape)
            # style_ref2 = writer_samples_list[1]
            
            # If requested, display reference samples
            if args.show_references:
                fig = display_writer_samples(wid, writer_samples_list[:4], title=f"Writer ID: {wid} - Reference Samples")
                ref_path = output_dir / f"writer_{wid}_references.png"
                fig.savefig(ref_path)
                plt.close(fig)
            
            # Prepare style references for the model
            # max_w = max(style_ref1.shape[-1], style_ref2.shape[-1])
            # style_images = torch.ones(2, 64, max_w, dtype=torch.float32, device=device)
            # style_images[0, :, :style_ref1.shape[-1]] = style_ref1.squeeze()
            # style_images[1, :, :style_ref2.shape[-1]] = style_ref2.squeeze()
            style_images = style_ref1.unsqueeze(0).to(device)
            
            # Prepare content
            content_img = prepare_content_tensor(args.text, device, args.char_width)
            
            # Calculate latent size based on text length
            text_len = len(args.text)
            latent_width = text_len * args.char_width // 8  # VAE downsamples by factor of 8
            
            # Initialize starting noise
            x0 = torch.randn(1, 4, 64//8, latent_width, device=device)
            
            # Wrap model with CFG guidance
            wrapped_model = WrappedModelCFG(
                unet, 
                cfg_scale=args.cfg
                # cfg_scale_text=args.text_cfg,
                # cfg_scale_style=args.style_cfg
            )
            
            # Setup ODE solver
            time_steps = torch.linspace(0, 1, args.steps).to(device)
            solver = ODESolver(wrapped_model)
            
            # Generate sample
            print(f"Generating '{args.text}' using writer {wid}...")
            output_path = output_dir / f"writer_{wid}_{args.text.replace(' ', '_')}.png"
            
            # Sample with ODE solver
            sol = solver.sample(
                x_init=x0,
                step_size=args.step_size,
                method=args.method,
                time_grid=time_steps,
                return_intermediates=True,
                content=content_img,
                style=style_images,
            )
            
            # Convert to image
            sol = sol.detach().cpu()
            img = sol[-1]
            latents = 1 / 0.18215 * img
            img = vae.decode(latents.to(device)).sample
            
            # Save individual image
            arr = img[0,0].detach().cpu().clamp(-1, 1)  
            arr = ((arr + 1) / 2 * 255).round().to(torch.uint8)
            # final_pil = Image.fromarray(
            #     ((final_image.numpy() + 1) / 2 * 255).astype(np.uint8)
            # )
            # arr = arr.squeeze(0)                                 # drop channel dim
            final_pil = Image.fromarray(arr.numpy(), mode="L")

            final_pil = ImageOps.invert(final_pil) 
            final_pil.save(output_path)
            print(f"Saved to {output_path}")
            
            # Store for grid
            generated_images.append(final_pil)
            writer_labels.append(f"Writer {wid}")
    
    # Create grid if requested
    # if args.create_grid and len(generated_images) > 1:
    #     from batch_inference_script import create_grid_image
        
    #     grid_path = output_dir / f"{args.text.replace(' ', '_')}_grid.png"
    #     grid_img = create_grid_image(
    #         generated_images, 
    #         writer_labels, 
    #         cols=min(4, len(generated_images)),
    #         cell_height=200
    #     )
    #     grid_img.save(grid_path)
    #     print(f"Grid image saved to {grid_path}")
    
    print(f"All images saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()