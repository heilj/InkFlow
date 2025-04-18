import os
import argparse
import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm 

# ---------------------------- Transforms & Model ---------------------------- #

def init_models(device):
    vit_model = vit_b_16()
    checkpoint = torch.load("/scratch/rw3239/FontDiffuser/vit_b_16-c867db91.pth", map_location="cpu")
    vit_model.load_state_dict(checkpoint)
    vit_model.to(device)
    vit_model.eval()
    feature_extractor = create_feature_extractor(vit_model, return_nodes={"encoder.ln": "features"})
    return feature_extractor

dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

pixelwise_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])


# ---------------------------- Metric Functions ---------------------------- #

def load_image(path, size=(256, 256)):
    img = Image.open(path).convert("RGB").resize(size)
    return transforms.ToTensor()(img)


def compute_rmse(img1, img2):
    return torch.sqrt(torch.mean((img1 - img2) ** 2)).item()


def compute_l1(img1, img2):
    return torch.mean(torch.abs(img1 - img2)).item()


def compute_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).numpy()
    img2_np = img2.permute(1, 2, 0).numpy()
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=-1)


def compute_psnr(img1, img2):
    img1_np = img1.permute(1, 2, 0).numpy()
    img2_np = img2.permute(1, 2, 0).numpy()
    return psnr(img1_np, img2_np, data_range=1.0)


def compute_fid_from_files(files1, files2, device):
    fid = FrechetInceptionDistance(
        feature=2048,
        feature_extractor_weights_path="/scratch/rw3239/FontDiffuser/weights-inception-2015-12-05-6726825d.pth"
    ).to(device)
    for f in tqdm(files1, desc="FID real"):
        img = load_image(f)
        img = img.mul(255).clamp(0, 255).to(torch.uint8).unsqueeze(0).to(device)
        fid.update(img, real=True)
    for f in tqdm(files2, desc="FID fake"):
        img = load_image(f)
        img = img.mul(255).clamp(0, 255).to(torch.uint8).unsqueeze(0).to(device)
        fid.update(img, real=False)
    return fid.compute().item()


def compute_lpips(images1, images2, device):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    return lpips(images1, images2).mean().item()

def compute_lpips_from_files(files1, files2, device):
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    total = 0.0
    for f1, f2 in tqdm(zip(files1, files2), total=len(files1), desc="LPIPS"):
        img1 = load_image(f1).unsqueeze(0).to(device)  # [1, 3, H, W]
        img2 = load_image(f2).unsqueeze(0).to(device)
        score = lpips_metric(img1, img2).item()
        total += score
    return total / len(files1)


def compute_dino_score(img1: Image.Image, img2: Image.Image, feature_extractor, device) -> float:
    with torch.no_grad():
        tensor1 = dino_transform(img1).unsqueeze(0).to(device)
        tensor2 = dino_transform(img2).unsqueeze(0).to(device)

        feat1 = feature_extractor(tensor1)["features"].squeeze(0)[0]
        feat2 = feature_extractor(tensor2)["features"].squeeze(0)[0]

        cos_sim = F.cosine_similarity(feat1, feat2, dim=0).item()
    return cos_sim


# ---------------------------- Evaluation Logic ---------------------------- #

def evaluate_single(file1, file2, args):
    img1_tensor = load_image(file1).unsqueeze(0)
    img2_tensor = load_image(file2).unsqueeze(0)
    img1_pil = Image.open(file1).convert("RGB")
    img2_pil = Image.open(file2).convert("RGB")

    results = {}
    if args.rmse: results['RMSE'] = compute_rmse(img1_tensor[0], img2_tensor[0])
    if args.l1: results['L1'] = compute_l1(img1_tensor[0], img2_tensor[0])
    if args.ssim: results['SSIM'] = compute_ssim(img1_tensor[0], img2_tensor[0])
    if args.psnr: results['PSNR'] = compute_psnr(img1_tensor[0], img2_tensor[0])
    if args.lpips: results['LPIPS'] = compute_lpips(img1_tensor, img2_tensor)
    if args.fid: results['FID'] = None
    if args.dino: results['DINO'] = compute_dino_score(img1_pil, img2_pil)

    if args.fid:
        print("[Warning] Skipping FID in single image mode (requires >1 sample)")

    return results


def evaluate_folder(folder1, folder2, args, feature_extractor, device):

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    feature_extractor = init_models(device)

    files1 = sorted(glob(os.path.join(folder1, "*")))
    files2 = sorted(glob(os.path.join(folder2, "*")))

    assert len(files1) == len(files2), "Folders must contain the same number of images."

    total = {k: 0.0 for k in ['RMSE', 'L1', 'SSIM', 'PSNR', 'FID', 'LPIPS', 'DINO'] if getattr(args, k.lower())}
    n = len(files1)

    print(f"Evaluating {n} pairs of images...")
    print("Computing FID and LPIPS...")

    if args.lpips:
        total['LPIPS'] = compute_lpips_from_files(files1, files2, device)

    if args.fid:
        total['FID'] = compute_fid_from_files(files1, files2, device)
    
    print("Computing DINO...")

    if args.dino:
        for f1, f2 in tqdm(zip(files1, files2)):
            img1 = Image.open(f1).convert("RGB")
            img2 = Image.open(f2).convert("RGB")
            total['DINO'] += compute_dino_score(img1, img2, feature_extractor, device)
    
    print("Computing pixel-wise metrics...")

    for f1, f2 in tqdm(zip(files1, files2)):
        img1 = load_image(f1)
        img2 = load_image(f2)
        if args.rmse: total['RMSE'] += compute_rmse(img1, img2)
        if args.l1: total['L1'] += compute_l1(img1, img2)
        if args.ssim: total['SSIM'] += compute_ssim(img1, img2)
        if args.psnr: total['PSNR'] += compute_psnr(img1, img2)

    for k in total:
        if k not in ['FID', 'LPIPS']:
            total[k] /= n
    
    print(total)

    return total


# ---------------------------- Main Script ---------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image similarity metrics")

    parser.add_argument("input1", type=str, help="Path to first image or folder")
    parser.add_argument("input2", type=str, help="Path to second image or folder")
    parser.add_argument("--folder", action="store_true", help="Set this flag if inputs are folders")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration if available")

    # Individual metrics
    parser.add_argument("--rmse", action="store_true", default=True)
    parser.add_argument("--l1", action="store_true", default=True)
    parser.add_argument("--ssim", action="store_true", default=True)
    parser.add_argument("--psnr", action="store_true", default=True)
    parser.add_argument("--fid", action="store_true", default=True)
    parser.add_argument("--lpips", action="store_true", default=True)
    parser.add_argument("--dino", action="store_true", default=True)

    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    feature_extractor = init_models(device)

    if args.folder:
        results = evaluate_folder(args.input1, args.input2, args, feature_extractor, device)
    else:
        results = evaluate_single(args.input1, args.input2, args)

    for metric, value in results.items():
        print(f"{metric}: {value}")