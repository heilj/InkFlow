import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchdiffeq import odeint
import signal
import sys

from lib.datasets import Hdf5Dataset, BucketSampler, LengthAwareSubset
from lib.alphabet import strLabelConverter
from torch.utils.data import random_split
from lib.alphabet import get_lexicon, get_true_alphabet
from torch.optim.lr_scheduler import StepLR, LambdaLR
from networks.module import Recognizer  
from torch.nn import CTCLoss

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from torch import Tensor
from torch.amp import GradScaler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm as std_tqdm
from transformers import HfArgumentParser

# from flow_matching.models import UNetModel
from flow_matching.sampler import PathSampler
from flow_matching.solver import ModelWrapper, ODESolver
from flow_matching.utils import model_size_summary, set_seed

from networks.unet_cfg import UNetGenerator
# from networks.encoder import ContentStyleEncoder
from networks.utils import letter2index, con_symbols

from networks.module import Recognizer  
from networks.ema import EMA 
from diffusers import AutoencoderKL

# Import distributed training modules
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math

tqdm = partial(std_tqdm, dynamic_ncols=True)

# Set environment variables for distributed training
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"  # Choose any free port

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def skewed_timestep_sampling(batch_size, device):
    """Sample timesteps with more focus on difficult regions."""
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((batch_size,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    t = 1 / (1 + sigma)
    t = torch.clip(t, min=0.0001, max=1.0)
    return t

def print_learning_rates(optimizer, epoch):
    print(f"[Epoch {epoch}] Learning Rates:")
    for i, group in enumerate(optimizer.param_groups):
        print(f" - Group {i} LR: {group['lr']:.6f}")

def optimized_scale(positive, negative):
    """
    positive: v_cond, shape (B, *)
    negative: v_uncond, shape (B, *)
    Returns: s_star, shape (B, 1)
    """
    dot = torch.sum(positive * negative, dim=1, keepdim=True)           # (B, 1)
    norm_sq = torch.sum(negative ** 2, dim=1, keepdim=True) + 1e-8       # avoid div 0
    return dot / norm_sq  # (B, 1)

def get_lr_scheduler(optimizer, total_steps):
    """Create learning rate scheduler with warm-up and consistency after loading"""
    def lr_lambda(current_step):
        # Warm-up for 5% of training
        warmup_steps = int(0.01 * total_steps)
        
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

def freeze_content_style_encoder(unet_model):
    """Freeze the ContentStyleEncoder module within the UNet"""
    # Check if the UNet has the mix attribute
    if hasattr(unet_model, 'mix'):
        for param in unet_model.mix.parameters():
            param.requires_grad = False
        unet_model.mix.eval()
        for m in unet_model.mix.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
        print(f"ContentStyleEncoder 'mix' module frozen")
    else:
        print("Warning: Could not find ContentStyleEncoder ('mix') in UNet model")
    
    # Verify frozen state
    frozen_count = 0
    trainable_count = 0
    for name, param in unet_model.named_parameters():
        if not param.requires_grad:
            frozen_count += 1
        else:
            trainable_count += 1
    print(f"Frozen parameters: {frozen_count}, Trainable parameters: {trainable_count}")

def get_param_groups_with_encoder(unet_model, encoder_lr, unet_lr):
    """Create parameter groups with different learning rates"""
    encoder_params = []
    other_params = []
    
    # Separate ContentStyleEncoder parameters
    for name, param in unet_model.named_parameters():
        if 'mix.' in name:  # Parameters from the ContentStyleEncoder
            encoder_params.append(param)
        else:  # All other parameters
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': other_params, 'lr': unet_lr}
    ]
    
    print(f"ContentStyleEncoder parameters: {len(encoder_params)}")
    print(f"Other UNet parameters: {len(other_params)}")
    
    return param_groups

def setup(rank, world_size):
    """Initialize the distributed environment."""
    # Add these lines
    local_rank = os.environ.get("LOCAL_RANK", rank)
    torch.cuda.set_device(local_rank)
    # Then modify your init_process_group call
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.size(0)).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        return self.model(x_t=x, t=t, **extras)

@dataclass
class ScriptArguments:
    do_train: bool = True
    do_sample: bool = False
    batch_size: int = 380
    n_epochs: int = 800
    encoder_lr: float = 0.00002  # Lower learning rate for the encoder (1/3 of the UNet rate)
    unet_lr: float = 0.00008
    sigma_min: float = 0.02
    seed: int = 1001
    output_dir: str = "outputs"
    num_classes = 80
    char_width = 32
    lambda_ctc = 0.1
    starting_epoch = 600
    world_size: int = 2  # Number of GPUs to use
    freeze_encoder: bool = False
    preserve_lr: bool = False 
    cfg_train_ratio: float = 0.1
    eval_cfg_scales: str = "1.0,5.0,7.5,10.0" 
    encoder_decay_start_epoch: int = 400
    freeze_encoder_epoch: int = 450
    cfg_scale_text: float = 6 
    cfg_scale_style: float = 6
    cfg_scale: float = 3


def train_flow_matching_model(rank, world_size, args):
    """
    Trains the style-conditioned flow matching model on handwritten word images.
    Modified for distributed training.
    """
    # === Setup distributed environment ===
    setup(rank, world_size)

    try:
        
        # === Setup ===
        output_dir = Path(args.output_dir) / "FM_style_cfg_new"
        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(f"cuda:{rank}")
        set_seed(args.seed + rank)  # Ensure different seeds per process
        print(f"Process {rank}/{world_size} using device: {device}")

        # if not hasattr(args, 'cfg_scale'):
        #     args.cfg_scale = 3.5

        # === Dataset ===
        transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
        dataset = Hdf5Dataset(
            root='data/iam',
            split='trnvalset_words64_OrgSz.hdf5',
            transforms=transforms,
            alphabet_key='iam_word'
        )
        train_len = int(0.9 * len(dataset))
        val_len = len(dataset) - train_len
        # t_val_len = int(0.1 * val_len)

        train_set, val_set = random_split(dataset, [train_len, val_len])
        # train_set = LengthAwareSubset(dataset, train_set.indices)
        # val_set = LengthAwareSubset(dataset, val_set.indices)
        # _, t_val_set = random_split(val_set, [val_len - t_val_len, t_val_len])
                
        # Adjust batch size per GPU
        batch_size = args.batch_size // world_size
        if batch_size < 1:
            batch_size = 1
        # Create distributed sampler for the training dataset
        # sampler = BucketSampler(
        #     dataset=train_set,
        #     batch_size=batch_size,
        #     drop_last=True,
        #     # shuffle_batches=True,
        #     rank=rank,
        #     num_replicas=2

        # )
        train_sampler = DistributedSampler(
            train_set, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )


        # Use distributed sampler with DataLoader
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            # batch_sampler=sampler,  # Use sampler instead of shuffle
            sampler=train_sampler,
            num_workers=8,
            collate_fn=Hdf5Dataset.collect_fn,
            drop_last=True,
            pin_memory=True , # Speeds up data transfer to GPU
            persistent_workers=True
        )

        # For validation, we don't need to distribute
        if rank == 0:  # Only create validation loader on main process
            # batch_sampler = BucketSampler(
            #     num_buckets=5,
            #     dataset=val_set,
            #     batch_size=batch_size,
            #     drop_last=True,
            #     # shuffle_batches=True
            # )
            l_valid_loader = DataLoader(
                val_set,
                # batch_sampler=batch_sampler,
                batch_size=4,
                shuffle=False,
                num_workers=8,
                collate_fn=Hdf5Dataset.collect_fn,
                drop_last=True,
                persistent_workers=True
            )
            t_valid_loader = DataLoader(
                val_set,
                # batch_sampler=batch_sampler,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=Hdf5Dataset.collect_fn,
                drop_last=True,
                persistent_workers=True
            )
            wid_style_samples = {}
            for batch in tqdm(l_valid_loader, desc="Preloading style samples"):
            # Process each item in batch to handle potentially different writer IDs
                images = batch['org_imgs'].to(device)
                # print(images.shape)
                for i in range(len(batch['wids'])):
                    wid = int(batch['wids'][i])
                    if wid not in wid_style_samples:
                        wid_style_samples[wid] = []
                    # print(batch['org_imgs'][i].shape)
                    wid_style_samples[wid].append(batch['org_imgs'][i].clone())
            
            # Filter out writers with insufficient samples
            valid_wids = [wid for wid, samples in wid_style_samples.items() if len(samples) >= 2]
            invalid_wids = [wid for wid, samples in wid_style_samples.items() if len(samples) < 2]

            for wid in invalid_wids:
                del wid_style_samples[wid]
        
            print(f"Found {len(valid_wids)} writers with at least {2} samples")
            print(f"Excluded {len(invalid_wids)} writers with insufficient samples")
        
        # === Models ===
        # Load VAE
        vae = AutoencoderKL.from_pretrained('stable-diffusion-v1-5', subfolder="vae")
        vae.requires_grad_(False)
        vae = vae.to(device)

        # Initialize models
        # encoder = ContentStyleEncoder(d_model=512).to(device)
        unet = UNetGenerator(input_channels=4, base_channels=512, context_dim=512).to(device)
        
        ema_decay = 0.999  # High decay rate for stability
        
        # Loss function
        path_sampler = PathSampler(sigma_min=args.sigma_min)

        # encoder = DDP(encoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        # unet = DDP(unet, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        # unet_ema = EMA(unet.module, decay=ema_decay)
        

        unet_ddp = DDP(unet, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        unet_ema = EMA(model=unet_ddp.module, decay=ema_decay)
        
        # Optimizer and scheduler
        param_groups = get_param_groups_with_encoder(
            unet_ddp.module, 
            encoder_lr=args.encoder_lr,  # Lower learning rate for encoder
            unet_lr=args.unet_lr         # Regular learning rate for rest of UNet
        )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
        # total_steps = len(train_loader) * (args.n_epochs - args.starting_epoch)
        # scheduler = get_lr_scheduler(optimizer, args, total_steps)
        scaler = GradScaler(enabled=(device.type == 'cuda'))
        
        # Only load weights on the main process and broadcast to others
        if rank == 0 and args.starting_epoch > 0:
            ckpt_path = output_dir / f"epoch_{args.starting_epoch}.pth"
            checkpoint = torch.load(ckpt_path, map_location=device)
            # encoder.module.load_state_dict(checkpoint['encoder'])
            unet_ddp.module.load_state_dict(checkpoint['unet'])
            if 'unet_ema' in checkpoint:
                unet_ema.load_state_dict(checkpoint['unet_ema'])
            encoder_frozen = False
            if args.freeze_encoder:
                print(f"Freezing ContentStyleEncoder after loading checkpoint")
                freeze_content_style_encoder(unet_ddp.module)
                # encoder_frozen = True
                # param_groups = get_param_groups_with_encoder(
                #     unet_ddp.module, 
                #     encoder_lr=0.0,  # Zero learning rate for frozen encoder
                #     unet_lr=args.unet_lr
                # )
                # optimizer = torch.optim.AdamW(param_groups, weight_decay=0.002)

            else:
                # Set up regular parameter groups with different learning rates
                param_groups = get_param_groups_with_encoder(
                    unet_ddp.module, 
                    encoder_lr=args.encoder_lr,
                    unet_lr=args.unet_lr
                )
                
                optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
            
            # Load optimizer state, but preserve our learning rate settings
            optimizer_dict = checkpoint['optimizer']
            
            # Modify the optimizer state to use our learning rates
            # if not args.preserve_lr:
            for i, param_group in enumerate(optimizer_dict['param_groups']):
                if i == 0:  # ContentStyleEncoder parameters
                    param_group['lr'] = args.encoder_lr 
                else:  # Other parameters
                    param_group['lr'] = args.unet_lr
            
            # Load the possibly modified optimizer state
            optimizer.load_state_dict(optimizer_dict)
            # for g in optimizer.param_groups:
            #     g["weight_decay"] = 0.05
            # trainable_params = [p for p in unet_ddp.module.parameters() if p.requires_grad]
            # total_params = [p for p in unet_ddp.module.parameters()]
            # total_param_count = sum(p.numel() for p in unet_ddp.module.parameters())
            # lr_scale = len(trainable_params) / len(total_params)
            # new_lr = args.unet_lr * lr_scale          # simple linear heuristic
            # print(new_lr)

            # optimizer = torch.optim.AdamW(trainable_params, lr=new_lr,
            #                             weight_decay=1e-4) 
            scaler.load_state_dict(checkpoint['scaler'])

            # if 'scheduler' in checkpoint and hasattr(args, 'preserve_scheduler') and args.preserve_scheduler:
            #     scheduler.load_state_dict(checkpoint['scheduler'])
        
        total_steps = len(train_loader) * (args.n_epochs - args.starting_epoch)
        # scheduler = get_lr_scheduler(optimizer, total_steps)
        # if rank == 0 and args.starting_epoch > 0:
        #     if 'scheduler' in checkpoint and hasattr(args, 'preserve_scheduler') and args.preserve_scheduler:
        #         try:
        #             scheduler.load_state_dict(checkpoint['scheduler'])
        #             print("Successfully loaded scheduler state from checkpoint")
        #         except Exception as e:
        #             print(f"Warning: Failed to load scheduler state: {e}")
        #             print("Continuing with freshly initialized scheduler")


        # unet = EMA(model=unet, decay=ema_decay)
        # Wait for the main process
        dist.barrier()

        # Wrap models with DDdP after moving to GPU
        # encoder = encoder.to(device)
        # unet = unet.to(device)

        

        # if args.starting_epoch > 0 and args.freeze_encoder:
        #     # When freezing after DDP wrapping, you need to access the module
        #     for param in encoder.module.parameters():
        #         param.requires_grad = False
            
            # print("Encoder parameters frozen for continued training")
            
            # # Recreate optimizer with only UNet parameters
            # optimizer = torch.optim.AdamW([
            #     {'params': unet_ddp.module.parameters(), 'lr': args.unet_lr}
            # ], weight_decay=0.002)
            # print("Optimizer recreated with only UNet parameters")

        # Print model summary on main process only
        if rank == 0:
            model_size_summary(unet_ddp.module)
            print("GradScaler enabled:", scaler._enabled)

        # === Training Loop ===
        train_losses = []
        valid_losses = []
        valid_losses_text = []
        
        for epoch in range(args.starting_epoch, args.n_epochs):
            train_sampler.set_epoch(epoch)  # Important for proper shuffling in DistributedSampler
            
            unet_ddp.train()
            unet_ema.train(True)
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.n_epochs}") if rank == 0 else train_loader
            total_loss = 0
            batch_count = 0
            
            for batch in pbar:
                # === Load Data ===
                images = batch['org_imgs'].to(device)     
                # print(images.shape)        
                content_img = batch['content'].to(device) 
                style_img = batch['style_imgs'].to(device)
                
                images = images.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    images = vae.encode(images).latent_dist.sample()
                    images = images * 0.18215

                # === Process Content ===            
                
                # === Sample Flow Path ===
                x_1 = images
                x_0 = torch.randn_like(x_1)                                   
                t = torch.rand(x_1.size(0), 1, device=device) 
                # t = skewed_timestep_sampling(x_1.size(0), device)                
                x_t, dx_t = path_sampler.sample(x_0, x_1, t.squeeze(1))      

                # === Forward & Loss ===
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
                    pred_v = unet_ddp(x_t, content_img, style_img, t, tag='train')                     
                    loss_mse = F.mse_loss(pred_v, dx_t)
                
                loss = loss_mse

                # === Backprop ===
                scaler.scale(loss).backward()
                # for name, param in unet_ddp.named_parameters():
                #     if param.grad is None:
                #         print(name)
                # torch.nn.utils.clip_grad_norm_(param_groups, max_norm=5)
                for param_group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=1)
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                unet_ema.update_ema()

                if rank == 0 and hasattr(pbar, 'set_postfix'): 
                    pbar.set_postfix(loss=loss.item())
                
                total_loss += loss.item()
                batch_count += 1
                
            # Average loss across all processes
            # print(batch_count)
            avg_loss = torch.tensor([total_loss / batch_count], device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_train_loss = avg_loss.item() / world_size
            
            if rank == 0:

                # print_learning_rates(optimizer, epoch)

                train_losses.append(avg_train_loss)
                print(f"Epoch {epoch+1}/{args.n_epochs} | Train Loss: {avg_train_loss:.4f}")
                if epoch == args.starting_epoch or (epoch + 1) % 10 == 0:
                    # Validation on main process only

                    unet_ddp.eval()
                    unet_ema.train(False)
                    val_mse = validate(
                        unet_ema.model,
                        path_sampler, t_valid_loader, device, vae, cfg_scale=args.cfg_scale
                    )
                    # avg_valid_loss = val_mse + args.lambda_ctc * val_ctc
                    valid_losses.append(val_mse)
                    # valid_losses_text.append(val_mse_text)
                    print(f"Validation Loss: {val_mse:.4f} (MSE: {val_mse:.4f})")
                    plot_few_shot_style_transfer(args, epoch+1, unet_ema.model, vae, device, 
                                     wid_style_samples, cfg_scale=args.cfg_scale)
                    unet_ddp.train()
                    unet_ema.train(True)
                
                # === Save checkpoint ===
                if epoch == args.starting_epoch or (epoch + 1) % 100 == 0:
                    ckpt_path = output_dir / f"epoch_{epoch+1:02d}.pth"
                    torch.save({
                        # 'encoder': encoder.module.state_dict(),
                        'unet': unet_ddp.module.state_dict(),
                        'unet_ema': unet_ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(), 
                        'train_losses': train_losses,
                        'valid_losses': valid_losses,
                    }, ckpt_path)
                # if epoch == args.encoder_decay_start_epoch:
                #     print(f"[Epoch {epoch}] Starting encoder learning rate decay.")
                # elif epoch > args.encoder_decay_start_epoch and epoch < args.freeze_encoder_epoch:
                #     # Linear decay of encoder LR
                #     decay_frac = (args.freeze_encoder_epoch - epoch) / (args.freeze_encoder_epoch - args.encoder_decay_start_epoch)
                #     new_encoder_lr = args.encoder_lr * decay_frac
                #     optimizer.param_groups[0]['lr'] = new_encoder_lr  # Assumes encoder param group is at index 0
                #     print(f"[Epoch {epoch}] Decaying encoder LR to {new_encoder_lr:.6f}")
                # elif epoch == args.freeze_encoder_epoch:
                #     print(f"[Epoch {epoch}] Freezing encoder.")
                #     freeze_content_style_encoder(unet_ddp.module)
                #     optimizer.param_groups[0]['lr'] = 0.0
                    # Generate visualizations
                
            
            # Make sure all processes sync before continuing
            dist.barrier()
        
        # Clean up
    except Exception as e:
            print(f"Process {rank} encountered error: {e}")
            raise e
    
    finally:
        dist.destroy_process_group()


def validate(unet, path_sampler, val_loader, device, vae, cfg_scale=3):
    """Validation function focused on generation quality."""
    # unet.eval()
    # encoder.eval()


    total_mse_cond, total_ctc = 0.0, 0.0
    n_batches = len(val_loader)
    
    # Set specific evaluation points - more focused on end of trajectory
    # where generation quality matters most
    eval_t_values = [0.9, 0.95, 0.99]  # Example values close to x₁
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['org_imgs'].to(device)   
            content_img = batch['content'].to(device)
            style_img = batch['style_imgs'].to(device)

            # content_h = encoder(content_img, style_img)
            # cond_feat = content_h

            images = images.repeat(1, 3, 1, 1)
            images = vae.encode(images).latent_dist.sample()
            images = images * 0.18215

            x_1 = images
            
            batch_mse_cond, batch_mse_text = 0.0, 0.0
            
            # Evaluate at specific t values instead of random
            for t_val in eval_t_values:
                x_0 = torch.randn_like(x_1)
                t = torch.ones(x_1.size(0), device=device) * t_val
                
                x_t, dx_t = path_sampler.sample(x_0, x_1, t)
                
                v_cond, v_uncond = unet(x_t, content_img, style_img, t.unsqueeze(1),tag='sample', cfg_scale=cfg_scale)

                # v_cond_flat = v_cond.view(v_cond.size(0), -1)
                # v_uncond_flat = v_uncond.view(v_uncond.size(0), -1)
                # v_null_flat = v_null.view(v_null.size(0), -1)
                # # Compute optimized scale s*
                # s_star_base = optimized_scale(v_cond_flat, v_null_flat)   # (B, 1)
                # s_star_style = optimized_scale(v_uncond_flat, v_cond_flat)   # (B, 1)

                # Reshape for broadcasting
                # while s_star_base.dim() < v_uncond.dim():
                #     s_star_base = s_star_base.unsqueeze(-1)
                # while s_star_style.dim() < v_uncond.dim():
                #     s_star_style = s_star_style.unsqueeze(-1)

                # Apply CFG-Zero*
        #         if t[0].item() < 1e-5:  # Check if we're at the initial time step
        #             v_guided = torch.zeros_like(v_uncond)  # For initial step
        #         else:
        #             # Guided velocity with optimized scale
        #             v_guided = v_null * s_star_base + \
        #    cfg_scale_text * (v_uncond - v_null * s_star_base) + \
        #    cfg_scale_style * (v_cond - v_uncond * s_star_style)
        #         pred_v = v_guided
                loss_mse_cond = F.mse_loss(v_cond, dx_t)
                # loss_mse_text = F.mse_loss(v_uncond, dx_t)
                
                batch_mse_cond += loss_mse_cond.item() / len(eval_t_values)
                # batch_mse_text += loss_mse_text.item() / len(eval_t_values)
            
            total_mse_cond += batch_mse_cond
            # total_mse_text += batch_mse_text

    avg_mse_cond = total_mse_cond / n_batches
    # avg_mse_text = total_mse_text / n_batches
    return avg_mse_cond 

def plot_few_shot_style_transfer(args, epoch, unet, vae, device, wid_style_samples, num_samples=8, k=2, cfg_scale=3):
    os.makedirs("fm_style_cfg_new", exist_ok=True)

    # Load models
    # unet = unet.eval()
    # encoder = encoder.eval()

    lexicon = get_lexicon('./data/english_words.txt',
                        get_true_alphabet('iam_word_org'),
                        max_length=6)
    # print(list(wid_style_samples.keys()))
    selected_wids = random.sample(list(wid_style_samples.keys()), min(3, len(wid_style_samples)))
    selected_wids += [369, 75]
    fig, axes = plt.subplots(len(selected_wids), num_samples + k, figsize=(num_samples + k, 5))

    words = random.sample(lexicon, num_samples-4)
    words = words + ['Mr', 'could', 'light', 'size']
    content_imgs = []
    for word in words:
        content = [letter2index[i] for i in word]
        content_img = con_symbols[content]
        content_img = content_img.unsqueeze(0)
        content_imgs.append(content_img)

    alphabet = strLabelConverter('iam_word')
    fake_lbs, fake_lb_lens = alphabet.encode(words)
    fake_lb_lens = fake_lb_lens.to(device)

    for row, wid in enumerate(selected_wids):
        candidates = wid_style_samples[wid]
        if len(candidates) < k:
            continue

        selected = random.sample(candidates, k)

        max_w = max([style_image.shape[-1] for style_image in selected])
        # print(max_w)
        ref_imgs = [item[0].unsqueeze(0).to(device) for item in selected]
        new_style_images = torch.ones(2, 64, max_w, dtype=torch.float32, device=device)
        new_style_images[0, :, :selected[0].shape[-1]] = selected[0]
        new_style_images[1, :, :selected[1].shape[-1]] = selected[1]  # assume item[0] is image
        new_style_images = new_style_images.unsqueeze(0)
        # === Generate synthetic labels ===


        for j in range(k):
            axes[row, j].imshow(ref_imgs[j].squeeze().cpu(), cmap="gray")
            axes[row, j].set_title("Ref")
            axes[row, j].axis("off")

        for j in range(num_samples):
            # cond_feat = encoder.generate(content_imgs[j].to(device), new_style_images).to(device)

            label_len_j = fake_lb_lens[j].unsqueeze(0)
            # add = (label_len_j % 2).long()
            # # add the mask → +1 if odd, +0 if even
            # label_len_j = label_len_j + add
            # width = label_len_j.item() * args.char_width  # use your char_width from args
            x0 = torch.randn(1, 4, 64//8, label_len_j * args.char_width //8).to(device)
            
            flow = unet
            class WrappedModel(ModelWrapper):
                def forward(self, x, t, **extras):
                    
                    # ODE solver gives scalar t, but UNet expects shape [B, 1]
                    if t.dim() == 0:
                        t = t.expand(x.size(0)).unsqueeze(1)
                    elif t.dim() == 1:
                        t = t.unsqueeze(1)

                    content = extras["content"]
                    style = extras["style"]
                    
                    # Get conditional and unconditional predictions
                    v_cond, v_uncond = self.model(x_t=x, content=content, style=style, t=t, tag='sample', cfg_scale=cfg_scale)

                    # Flatten
                    v_cond_flat = v_cond.view(v_cond.size(0), -1)
                    v_uncond_flat = v_uncond.view(v_uncond.size(0), -1)
                    # v_null_flat = v_null.view(v_null.size(0), -1)
                    # Compute optimized scale s*
                    s_star_base = optimized_scale(v_cond_flat, v_uncond_flat)   # (B, 1)
                    # s_star_style = optimized_scale(v_uncond_flat, v_null_flat)   # (B, 1)

                    # Reshape for broadcasting
                    while s_star_base.dim() < v_uncond.dim():
                        s_star_base = s_star_base.unsqueeze(-1)
                    # while s_star_style.dim() < v_uncond.dim():
                    #     s_star_style = s_star_style.unsqueeze(-1)

                    # Apply CFG-Zero*
                    if t[0].item() < 1e-5:  # Check if we're at the initial time step
                        v_guided = torch.zeros_like(v_uncond)  # For initial step
                    else:
                        # Guided velocity with optimized scale
            #             v_guided = v_null * (s_star_base +  s_star_style)/2 + \
            # cfg_scale_text * (v_uncond - v_null * s_star_base) + \
            # cfg_scale_style * (v_cond - v_null * s_star_style)
                        v_guided = v_uncond * s_star_base + cfg_scale * (v_cond - v_uncond * s_star_base)
                    return v_guided
            
            wrapped_model = WrappedModel(flow)

            sample_steps = 101
            time_steps = torch.linspace(0, 1, sample_steps).to(device)
            step_size = 0.025
            solver = ODESolver(wrapped_model)
            sol = solver.sample(
                x_init=x0,
                step_size=step_size,
                method="midpoint",
                # method="dopri5",
                time_grid=time_steps,
                return_intermediates=True,
                content=content_imgs[j].to(device),
                style=new_style_images.to(device),
            )
            sol = sol.detach().cpu()
            img = sol[-1]
            latents = 1 / 0.18215 * img
            img = vae.decode(latents.to(device)).sample
            # img = (sol.clamp(-1, 1) + 1) / 2

            axes[row, j + k].imshow(img[0, 0].cpu(), cmap="gray")
            axes[row, j + k].set_title(words[j], fontsize=8)
            axes[row, j + k].axis("off")

    plt.tight_layout()
    vis_path = f"fm_style_cfg_new/epoch_{epoch}_fewshot.png"
    plt.savefig(vis_path)
    plt.close()
    print(f"Visualization saved to {vis_path}")



def evaluate_cfg_scales(args, epoch, unet, vae, device, wid_style_samples, val_loader=None):
    """
    Generate samples with different CFG scales to evaluate their effect
    """
    os.makedirs("fm_style_cfg_eval", exist_ok=True)
    
    # If no specific scales are provided, use defaults
    if not hasattr(args, 'eval_cfg_scales') or args.eval_cfg_scales is None:
        cfg_scales = [1.0, 3.0, 5.0, 7.5, 10.0]
    else:
        cfg_scales = args.eval_cfg_scales
    
    # Select a few representative samples for evaluation
    num_examples = 3  # Number of different content-style pairs to test
    num_scales = len(cfg_scales)
    
    # Create lexicon and prepare content
    lexicon = get_lexicon('./data/english_words.txt',
                         get_true_alphabet('iam_word_org'),
                         max_length=6)
    test_words = ['light', 'could', 'Mr']  # Fixed test words for consistency
    
    content_imgs = []
    for word in test_words:
        content = [letter2index[i] for i in word]
        content_img = con_symbols[content]
        content_img = content_img.unsqueeze(0)
        content_imgs.append(content_img)
    
    # Select a few representative writer IDs
    selected_wids = random.sample(list(wid_style_samples.keys()), 
                                  min(num_examples, len(wid_style_samples)))
    
    # Create a grid figure: rows for different content-style pairs, columns for different CFG scales
    fig, axes = plt.subplots(num_examples, num_scales + 1, figsize=(2*num_scales + 2, 2*num_examples))
    
    for i, wid in enumerate(selected_wids):
        # Get style references
        candidates = wid_style_samples[wid]
        if len(candidates) < 2:
            continue
            
        selected = random.sample(candidates, 2)
        max_w = max([style_image.shape[-1] for style_image in selected])
        
        # Prepare style image
        new_style_images = torch.ones(2, 64, max_w, dtype=torch.float32, device=device)
        new_style_images[0, :, :selected[0].shape[-1]] = selected[0]
        new_style_images[1, :, :selected[1].shape[-1]] = selected[1]
        new_style_images = new_style_images.unsqueeze(0)
        
        # Show style reference
        style_img = torch.cat([selected[0], selected[1]], dim=0)
        if num_examples == 1:
            axes[0].imshow(style_img.cpu(), cmap="gray")
            axes[0].set_title("Style Reference")
            axes[0].axis("off")
        else:
            axes[i, 0].imshow(style_img.cpu(), cmap="gray")
            axes[i, 0].set_title("Style Reference")
            axes[i, 0].axis("off")
        
        # Use a fixed content for consistency
        content_idx = i % len(test_words)
        content_img = content_imgs[content_idx].to(device)
        
        # Get content word
        word = test_words[content_idx]
        
        # Sample with different CFG scales
        for j, cfg_scale in enumerate(cfg_scales):
            # Prepare for ODE solving
            alphabet = strLabelConverter('iam_word')
            fake_lbs, fake_lb_lens = alphabet.encode([word])
            fake_lb_lens = fake_lb_lens.to(device)
            
            label_len_j = fake_lb_lens[0].unsqueeze(0)
            x0 = torch.randn(1, 4, 64//8, label_len_j * args.char_width //8).to(device)
            
            flow = unet
            class WrappedModel(ModelWrapper):
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    if t.dim() == 0:
                        t = t.expand(x.size(0)).unsqueeze(1)
                    elif t.dim() == 1:
                        t = t.unsqueeze(1)
                    return self.model(x_t=x, t=t, tag='sample', cfg_scale=cfg_scale, **extras)
            
            wrapped_model = WrappedModel(flow)
            
            # Sampling
            sample_steps = 201
            time_steps = torch.linspace(0, 1, sample_steps).to(device)
            step_size = 0.01
            solver = ODESolver(wrapped_model)
            
            sol = solver.sample(
                x_init=x0,
                step_size=step_size,
                method="dopri5",
                time_grid=time_steps,
                return_intermediates=True,
                content=content_img,
                style=new_style_images.to(device),
            )
            
            sol = sol.detach().cpu()
            img = sol[-1]
            latents = 1 / 0.18215 * img
            img = vae.decode(latents.to(device)).sample
            
            # Plot the result
            col_idx = j + 1  # +1 because column 0 is for style reference
            if num_examples == 1:
                axes[col_idx].imshow(img[0, 0].cpu(), cmap="gray")
                axes[col_idx].set_title(f"CFG={cfg_scale}")
                axes[col_idx].axis("off")
            else:
                axes[i, col_idx].imshow(img[0, 0].cpu(), cmap="gray")
                axes[i, col_idx].set_title(f"CFG={cfg_scale}")
                axes[i, col_idx].axis("off")
    
    plt.tight_layout()
    vis_path = f"fm_style_cfg_eval/epoch_{epoch}_cfg_comparison.png"
    plt.savefig(vis_path)
    plt.close()
    print(f"CFG comparison visualization saved to {vis_path}")

def main():

    def signal_handler(sig, frame):
        print('Ctrl+C detected, cleaning up...')
        # Make sure all processes are terminated
        if torch.distributed.is_initialized():
            cleanup()
        # You might also want to force kill any lingering processes
        print("Terminating all Python processes...")
        os.system("pkill -9 python")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    try:
        parser = HfArgumentParser(ScriptArguments)
        script_args, *_ = parser.parse_args_into_dataclasses()

        if hasattr(script_args, 'eval_cfg_scales'):
            script_args.eval_cfg_scales = [float(x) for x in script_args.eval_cfg_scales.split(',')]
        
        # Auto-detect number of available GPUs if not specified
        if script_args.world_size == -1:
            script_args.world_size = torch.cuda.device_count()
        
        print(f"Using {script_args.world_size} GPUs for distributed training")
        
        if script_args.world_size <= 1:
            print("No multiple GPUs detected or specified. Running on single GPU.")
            # Run without distributed setup
            script_args.world_size = 1
            train_flow_matching_model(0, 1, script_args)
        else:
            # Launch multiple processes for distributed training
            mp.spawn(
                train_flow_matching_model,
                args=(script_args.world_size, script_args),
                nprocs=script_args.world_size,
                join=True
            )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Additional cleanup if needed at the main process level
        print("Main process exiting")


if __name__ == "__main__":
    main()