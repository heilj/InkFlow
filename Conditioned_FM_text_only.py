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

from lib.datasets import Hdf5Dataset
from lib.alphabet import strLabelConverter
from torch.utils.data import random_split
from lib.alphabet import get_lexicon, get_true_alphabet
from torch.optim.lr_scheduler import StepLR
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

from networks.c_unet import UNetGenerator
from networks.encoder import StyleEncoder, ContentOnlyEncoder, CrossAttnFuser, ContentDecoderTR
from networks.utils import letter2index, con_symbols

from networks.module import Recognizer  

tqdm = partial(std_tqdm, dynamic_ncols=True)


# ========== Dataset ==========
transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
dataset = Hdf5Dataset(
    root='data/iam',
    split='trnvalset_words64_OrgSz.hdf5',
    transforms=transforms,
    alphabet_key='iam_word',
    process_style=True
)

train_len = int(0.01 * len(dataset))
val_len = len(dataset) - train_len

t_val_len = int(0.01 * val_len)

train_set, val_set = random_split(dataset, [train_len, val_len])
_, t_val_set = random_split(val_set, [val_len - t_val_len, t_val_len])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True,
                          collate_fn=Hdf5Dataset.sorted_collect_fn_for_ctc, drop_last=True)

wid_style_samples = {}
valid_loader = DataLoader(val_set, batch_size=1, shuffle=True,
                          collate_fn=Hdf5Dataset.sorted_collect_fn_for_ctc, drop_last=True)

t_valid_loader = DataLoader(t_val_set, batch_size=4, shuffle=True,
                          collate_fn=Hdf5Dataset.sorted_collect_fn_for_ctc, drop_last=True)

# style_encoder = StyleEncoder()
content_encoder = ContentOnlyEncoder()
# Choose fusion strategy: either concatenation or cross-attention
use_concat = True  # set False to use cross-attention variant
# fuser = CrossAttnFuser(embed_dim=256, num_heads=4)
unet = UNetGenerator(input_channels=1, cond_channels=256, base_channels=128)

for batch in tqdm(train_loader, desc="Preloading style refs"):
    wid = int(batch['wids'][0])
    if wid not in wid_style_samples:
        wid_style_samples[wid] = []
    wid_style_samples[wid].append((
        batch['org_imgs'][0].clone(),
    ))

lexicon = get_lexicon('./data/english_words.txt',
                      get_true_alphabet('iam_word_org'),
                      max_length=6)

alphabet = strLabelConverter('iam_word')


@dataclass
class ScriptArguments:
    do_train: bool = True
    do_sample: bool = False
    batch_size: int = 4
    n_epochs: int = 20
    learning_rate: float = 5e-4
    sigma_min: float = 0.0
    seed: int = 42
    output_dir: str = "outputs"
    num_classes = 80
    train_loader = train_loader
    valid_loader = t_valid_loader
    content_encoder = content_encoder
    unet = unet
    char_width = 32
    lambda_ctc = 0.1
    starting_epoch = 0

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# ========== Pretrained Recognizer from HiGan+ ==========
recognizer = Recognizer(
    n_class=80,           # or however many characters your alphabet has
    resolution=16,
    max_dim=256,
    in_channel=1,
    norm='bn',            # enable batch norm
    init='N02',
    rnn_depth=2,          # or 1 if that's how it was trained
    bidirectional=True
).to(device)
recognizer.load_state_dict(torch.load("./pretrained/ocr_iam_new.pth", map_location=torch.device("mps"))["Recognizer"])

recognizer.eval()

for param in recognizer.parameters():
    param.requires_grad = False

ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.size(0)).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        return self.model(x_t=x, t=t, **extras)


def train_flow_matching_model(args: ScriptArguments):
    """
    Trains the style-conditioned flow matching model on handwritten word images.
    Uses StyleEncoder, ContentEncoder, FusionModule, UNetGenerator, and PathSampler.
    """

    # === Setup ===
    output_dir = Path(args.output_dir) / "FM"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    set_seed(args.seed)
    print(f"Using device: {device}")

    # Move models to device
    # content_decoder = args.content_decoder.to(device)
    content_encoder = args.content_encoder.to(device)
    unet = args.unet.to(device)

    path_sampler = PathSampler(sigma_min=args.sigma_min)
    # Combine model parameters
    params = list(content_encoder.parameters()) + \
             list(unet.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scaler = GradScaler(enabled=(device.type == device))

    # Print model summary
    model_size_summary(unet)
    print("GradScaler enabled:", scaler._enabled)

    # === Training Loop ===
    train_losses, valid_losses = [], []
    for epoch in range(args.starting_epoch, args.n_epochs):
        unet.train()
        # content_decoder.train()
        content_encoder.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.n_epochs}")
        total_loss = 0
        for batch in pbar:
            # === Load Data ===
            x_1 = batch['org_imgs'].to(device)             # ground-truth handwriting
            # style_ref = batch['style_imgs'].to(device)
            label_seqs = batch['lbs'].to(device)
            label_lens = batch['lb_lens'].to(device)
            content_img = batch['content'].to(device) # rendered text img B x T x H x W
            img_lens = batch['org_img_lens'].to(device)
            # style_img = batch['style_imgs'].to(device)     # reference handwriting (1-shot style)

            # === Process Style ===
            # style_edge = laplacian_filter(style_img)                      # Bx1xHxW
            # style_input = torch.cat([style_img, style_edge], dim=1)      # Bx2xHxW
            # style_input = style_ref      # Bx2xHxW
            # style_emb = style_encoder(style_input)                       # B x D
            # === Process Content ===
            content_feat = content_encoder(content_img)                  # B x T x D
            # content_feat = content_decoder(content_h)                   # B x T x D
            # === Fuse ===
            cond_feat = content_feat                 # B x T x D
            
            # === Sample Flow Path ===
            x_0 = torch.randn_like(x_1)                                   # noise
            t = torch.rand(x_1.size(0), 1, device=device)                 # time [B,1]
            x_t, dx_t = path_sampler.sample(x_0, x_1, t.squeeze(1))      # [B,1,H,W], [B,1,H,W]

            # === Forward & Loss ===
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
                pred_v = unet(x_t, cond_feat, t)                         # predict velocity
                loss_mse = F.mse_loss(pred_v, dx_t)
            loss = loss_mse
    #         # --- Compute image ---
                
            
    #         wrapped_model = WrappedModel(unet)
    #         solver = ODESolver(wrapped_model)

    #         # Solve from x_t to x_1 using the same logic
    #         with torch.no_grad():
    #             sample_steps = 101
    #             time_steps = torch.linspace(0, 1, sample_steps).to(device)
    #             step_size = 0.05

    #             vt_pred = solver.sample(
    #                 x_init=x_t,  # [B, 1, H, W]
    #                 step_size=step_size,
    #                 method="midpoint",  # or "euler", "rk4", etc.
    #                 time_grid=time_steps,
    #                 return_intermediates=False,  # only final prediction
    #                 cond_feat=cond_feat
    # )
    #         # --- Compute CTC loss ---
    #         with torch.no_grad():  # recognizer is frozen
                
    #             logits = recognizer(vt_pred, img_lens)  # [B, T, C] -> T=timesteps, C=num_classes


           
    #         logits = logits.log_softmax(2).transpose(0, 1)  # [T, B, C]

    #         ctc_loss = ctc_loss_fn(
    #             logits,                # [T, B, C]
    #             label_seqs,            # [B, L]
    #             (img_lens // recognizer.len_scale),  # predicted lengths
    #             label_lens             # target lengths
    #         )

    #         loss = loss_mse + args.lambda_ctc * ctc_loss

            # === Backprop ===
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=loss.item())
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_mse, val_ctc = validate(
            unet, content_encoder,
            path_sampler, recognizer, args.valid_loader, device, args
        )
        avg_valid_loss = val_mse + args.lambda_ctc * val_ctc
        valid_losses.append(avg_valid_loss)

        print(f"Epoch {epoch+1}/{args.n_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_valid_loss:.4f}")

        # === Save checkpoint ===
        if epoch == 0 or (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"epoch_{epoch+1:02d}.pth"
            torch.save({
                'content_encoder': content_encoder.state_dict(),
                'unet': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            plot_few_shot_style_transfer(args,epoch+1,)


def validate(unet, content_encoder, path_sampler, recognizer, val_loader, device, args):
    unet.eval()
    content_encoder.eval()

    total_mse, total_ctc = 0.0, 0.0
    n_batches = len(val_loader)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            x_1 = batch['org_imgs'].to(device)
            # style_ref = batch['style_imgs'].to(device)
            label_seqs = batch['lbs'].to(device)
            label_lens = batch['lb_lens'].to(device)
            content_img = batch['content'].to(device)
            img_lens = batch['org_img_lens'].to(device)


            content_h = content_encoder(content_img)
            cond_feat = content_h

            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.size(0), 1, device=device)
            x_t, dx_t = path_sampler.sample(x_0, x_1, t.squeeze(1))

            pred_v = unet(x_t, cond_feat, t)
            loss_mse = F.mse_loss(pred_v, dx_t)

            # wrapped_model = WrappedModel(unet)
            # solver = ODESolver(wrapped_model)
            # sample_steps = 101
            # time_steps = torch.linspace(0, 1, sample_steps).to(device)

            # vt_pred = solver.sample(
            #     x_init=x_t,
            #     step_size=0.05,
            #     method="midpoint",
            #     time_grid=time_steps,
            #     return_intermediates=False,
            #     cond_feat=cond_feat
            # )

            # logits = recognizer(vt_pred, img_lens)
            # logits = logits.log_softmax(2).transpose(0, 1)

            # ctc_loss = ctc_loss_fn(
            #     logits,
            #     label_seqs,
            #     (img_lens // recognizer.len_scale),
            #     label_lens
            # )

            total_mse += loss_mse.item()
            # total_ctc += ctc_loss.item()

    avg_mse = total_mse / n_batches
    avg_ctc = total_ctc / n_batches
    return avg_mse, avg_ctc

def plot_few_shot_style_transfer(args, epoch, num_samples=5, k=2):
    os.makedirs("fm_vis", exist_ok=True)

    # Load models
    unet = args.unet.eval()
    content_encoder = args.content_encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    selected_wids = random.sample(list(wid_style_samples.keys()), min(5, len(wid_style_samples)))
    fig, axes = plt.subplots(len(selected_wids), num_samples + k, figsize=(num_samples + k, 5))

    words = random.sample(lexicon, num_samples)
    content_imgs = []
    for word in words:
        content = [letter2index[i] for i in word]
        content_img = con_symbols[content]
        content_img = content_img.unsqueeze(0)
        content_imgs.append(content_img)

    fake_lbs, fake_lb_lens = alphabet.encode(words)
    fake_lbs = fake_lbs.to(device)
    fake_lb_lens = fake_lb_lens.to(device)

    contnet_vecs = [content_encoder(img.to(device)) for img in content_imgs]


    for row, wid in enumerate(selected_wids):
        candidates = wid_style_samples[wid]
        if len(candidates) < k:
            continue

        selected = random.sample(candidates, k)
        ref_imgs = [item[0].unsqueeze(0).to(device) for item in selected]  # assume item[0] is image

        # === Generate synthetic labels ===
        


        # Encode style vectors
        # style_vecs = [style_encoder(img.to(device)) for img in ref_imgs]
        
        # style = torch.stack(style_vecs).mean(dim=0)  # [D]

        
        t_eval = torch.tensor([0.0, 1.0], device=device)

        for j in range(k):
            axes[row, j].imshow(ref_imgs[j].squeeze().cpu(), cmap="gray")
            axes[row, j].set_title("Ref")
            axes[row, j].axis("off")

        for j in range(num_samples):
            cond_feat = contnet_vecs[j].to(device)

            label_len_j = fake_lb_lens[j].unsqueeze(0)
            width = label_len_j.item() * args.char_width  # use your char_width from args

            x0 = torch.randn(1, 1, 64, width).to(device)

            # def ode_func(t, x):
            #     t_exp = t.expand(x.size(0))
            #     # style_exp = style.expand(x.size(0), -1)
            #     return unet(x, cond_feat, t_exp)
            
            flow = unet
            class WrappedModel(ModelWrapper):
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    # ODE solver gives scalar t, but UNet expects shape [B, 1]
                    if t.dim() == 0:  # e.g., tensor(0.5)
                        t = t.expand(x.size(0)).unsqueeze(1)  # → [B, 1]
                    elif t.dim() == 1:
                        t = t.unsqueeze(1)  # make sure it’s [B, 1]
                    return self.model(x_t=x, t=t, **extras)
            wrapped_model = WrappedModel(flow)

            sample_steps = 101
            time_steps = torch.linspace(0, 1, sample_steps).to(device)
            step_size = 0.05
            solver = ODESolver(wrapped_model)
            sol = solver.sample(
                x_init=x0,
                step_size=step_size,
                method="midpoint",
                time_grid=time_steps,
                return_intermediates=True,
                cond_feat=cond_feat,
            )
            sol = sol.detach().cpu()
            img = sol[-1]
            # img = (sol.clamp(-1, 1) + 1) / 2

            axes[row, j + k].imshow(img[0, 0].cpu(), cmap="gray")
            axes[row, j + k].set_title(words[j], fontsize=8)
            axes[row, j + k].axis("off")

    plt.tight_layout()
    vis_path = f"fm_vis/epoch_{epoch}_fewshot.png"
    plt.savefig(vis_path)
    plt.close()
    print(f"Visualization saved to {vis_path}")




if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, *_ = parser.parse_args_into_dataclasses()

    if script_args.do_train:
        train_flow_matching_model(script_args)
