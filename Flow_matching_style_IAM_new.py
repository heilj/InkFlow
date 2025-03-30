# Complete Flow Matching Model with HiGAN-style Style Encoder (using img_lens)

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


# ========== Config ==========
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
batch_size = 4
epochs = 60
lr = 5e-4
style_dim = 64
char_width = 16
num_classes = 80

# ========== Dataset ==========
transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
dataset = Hdf5Dataset(
    root='data/iam',
    split='trnvalset_words64_OrgSz.hdf5',
    transforms=transforms,
    alphabet_key='iam_word',
    process_style=True
)

train_len = int(0.4 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          collate_fn=Hdf5Dataset.collect_fn, drop_last=True)

wid_style_samples = {}
valid_loader = DataLoader(val_set, batch_size=1, shuffle=True,
                          collate_fn=Hdf5Dataset.collect_fn, drop_last=True)

for batch in tqdm(valid_loader, desc="Preloading style refs"):
    wid = int(batch['wids'][0])
    if wid not in wid_style_samples:
        wid_style_samples[wid] = []
    wid_style_samples[wid].append((
        batch['style_imgs'][0].clone(),     # clone to avoid in-place issues
        batch['style_img_lens'][0].item()
    ))

# lexicon = get_lexicon('iam_word')
lexicon = get_lexicon('./data/english_words.txt',
                                   get_true_alphabet('iam_word_org'),
                                   max_length=20)
print(random.sample(lexicon, 2))
# ========== Style Encoder ==========
class StyleEncoder(nn.Module):
    def __init__(self, style_dim=64, in_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU()
        )
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU()
        )
        self.mu = nn.Linear(in_dim, style_dim)

    def forward(self, x, img_lens):
        feat = self.backbone(x)  # (B, C, H, W)
        B, C, H, W = feat.shape
        valid_w = (img_lens // 4).clamp(min=1)  # downscale factor=4
        pooled = []
        for i in range(B):
            valid_feat = feat[i, :, :, :valid_w[i]]
            pooled_feat = valid_feat.mean(dim=[1, 2])  # mean over H and valid W
            pooled.append(pooled_feat)
        pooled = torch.stack(pooled)
        h = self.linear_style(pooled)
        return F.normalize(self.mu(h), dim=1)

# ========== Conditional U-Net ==========
class ConditionedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels + cond_dim, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, cond):
        x = F.silu(self.norm1(self.conv1(x)))
        cond = cond.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond], dim=1)
        return F.silu(self.norm2(self.conv2(x)))

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConditionedDoubleConv(in_channels, out_channels, cond_dim)

    def forward(self, x, cond):
        return self.conv(self.pool(x), cond)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConditionedDoubleConv(in_channels, out_channels, cond_dim)

    def forward(self, x1, x2, cond):
        x1 = self.up(x1)
        x1 = F.pad(x1, [0, x2.size(3) - x1.size(3), 0, x2.size(2) - x1.size(2)])
        return self.conv(torch.cat([x2, x1], dim=1), cond)

class ConditionalUNet(nn.Module):
    def __init__(self, style_dim=64):
        super().__init__()
        self.t_dim = 16
        self.len_dim = 8
        self.cond_dim = self.t_dim + self.len_dim + style_dim

        self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, self.t_dim))
        self.len_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, self.len_dim))

        self.inc = ConditionedDoubleConv(1, 64, self.cond_dim)
        self.down1 = Down(64, 128, self.cond_dim)
        self.down2 = Down(128, 256, self.cond_dim)
        self.up1 = Up(256 + 128, 128, self.cond_dim)
        self.up2 = Up(128 + 64, 64, self.cond_dim)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x, t, style, label_lens):
        t_emb = self.time_embed(t.view(-1, 1))
        len_emb = self.len_embed(label_lens.view(-1, 1).float() / 20.0)
        cond = torch.cat([t_emb, style, len_emb], dim=1).unsqueeze(-1).unsqueeze(-1)

        x1 = self.inc(x, cond)
        x2 = self.down1(x1, cond)
        x3 = self.down2(x2, cond)
        x = self.up1(x3, x2, cond)
        x = self.up2(x, x1, cond)
        return self.outc(x)

# ========== Initialize ==========
alphabet = strLabelConverter('iam_word')
style_encoder = StyleEncoder(style_dim).to(device)
model = ConditionalUNet(style_dim).to(device)
params = list(model.parameters()) + list(style_encoder.parameters())
optimizer = torch.optim.AdamW(params, lr=lr)

# ========== Training ==========
def train():
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            imgs = batch['org_imgs'].to(device)
            labels = batch['lbs'].to(device)
            label_lens = batch['lb_lens'].to(device)
            ref_imgs = batch['style_imgs'].to(device)
            ref_lens = batch['style_img_lens'].to(device)

            style = style_encoder(ref_imgs, ref_lens)
            noise = torch.randn_like(imgs)
            t = torch.rand(imgs.size(0), device=device)
            xt = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * imgs

            vt_pred = model(xt, t, style, label_lens)
            velocity_gt = imgs - noise
            loss = F.l1_loss(vt_pred, velocity_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

        if epoch == 0 or (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"ckpt/fm_model_epoch{epoch+1}.pth")
            torch.save(style_encoder.state_dict(), f"ckpt/style_encoder_epoch{epoch+1}.pth")
            torch.save(optimizer.state_dict(), f"ckpt/optimizer_epoch{epoch+1}.pth")
            plot_few_shot_style_transfer(epoch+1)

# ========== Visualization ==========
def plot_few_shot_style_transfer(epoch, num_samples=8, k=2):
    os.makedirs("fm_vis", exist_ok=True)
    model.eval()
    style_encoder.eval()

    selected_wids = random.sample(list(wid_style_samples.keys()), min(5, len(wid_style_samples)))
    fig, axes = plt.subplots(len(selected_wids), num_samples + k, figsize=(num_samples + k, 5))

    for row, wid in enumerate(selected_wids):
        candidates = wid_style_samples[wid]
        if len(candidates) < k:
            continue  # skip if not enough refs

        selected = random.sample(candidates, k)
        ref_imgs = [item[0] for item in selected]
        ref_lens = [item[1] for item in selected]

        max_w = max(img.shape[-1] for img in ref_imgs)
        padded_refs = [F.pad(img, (0, max_w - img.shape[-1]), value=-1.0) for img in ref_imgs]
        ref_imgs = torch.stack(padded_refs).to(device)
        ref_lens = torch.tensor(ref_lens).to(device)
        style = style_encoder(ref_imgs, ref_lens).mean(dim=0, keepdim=True)

        words = random.sample(lexicon, num_samples)
        fake_lbs, fake_lb_lens = alphabet.encode(words)
        fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(device)

        width = fake_lb_lens.max().item() * char_width
        x0 = torch.randn(num_samples, 1, 64, width).to(device)
        t_eval = torch.tensor([0.0, 1.0], device=device)

        def ode_func(t, x):
            t_exp = t.expand(x.size(0))
            return model(x, t_exp, style.expand(x.size(0), -1), fake_lb_lens)

        with torch.no_grad():
            generated = odeint(
                ode_func, 
                x0, 
                t_eval, 
                rtol=1e-5, 
                atol=1e-6, 
                method='dopri5',
                options={'dtype': torch.float32})[-1]
            
            gen_imgs = (generated.clamp(-1, 1) + 1) / 2

        for j in range(k):
            axes[row, j].imshow(ref_imgs[j].squeeze().cpu(), cmap="gray")
            axes[row, j].set_title("Ref")
            axes[row, j].axis("off")

        for j in range(num_samples):
            axes[row, j + k].imshow(gen_imgs[j].squeeze().cpu(), cmap="gray")
            axes[row, j + k].set_title(words[j], fontsize=8)
            axes[row, j + k].axis("off")

    plt.tight_layout()
    plt.savefig(f"fm_vis/epoch_{epoch}_fewshot.png")
    plt.close()


# ========== Run ==========
if __name__ == "__main__":
    os.makedirs("ckpt", exist_ok=True)
    train()
