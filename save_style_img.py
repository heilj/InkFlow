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
import cv2

transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
dataset = Hdf5Dataset(
    root='data/iam',
    split='trnvalset_words64_OrgSz.hdf5',
    transforms=transforms,
    alphabet_key='iam_word',
    process_style=True
)

wid_style_samples = {}
loader = DataLoader(dataset, batch_size=1, shuffle=True,
                          collate_fn=Hdf5Dataset.sorted_collect_fn_for_ctc, drop_last=True)

def show_img(img):
    image_tensor = img
    image = image_tensor.squeeze().numpy()  # shape: [H, W]
    # image = image * 0.5 + 0.5  # reverse normalization ([-1,1] → [0,1])

    # Show image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

for batch in tqdm(loader, desc="Preloading style refs"):
    wid = int(batch['wids'][0])
    if wid not in wid_style_samples:
        wid_style_samples[wid] = []
    wid_style_samples[wid].append((
        batch['org_imgs'][0].clone(),
    ))
    # show_img(batch['org_imgs'][0].clone())
    # break

style_base_path = 'data/style_wids'  # This becomes self.style_path
os.makedirs(style_base_path, exist_ok=True)

for wid, style_list in wid_style_samples.items():
    writer_dir = os.path.join(style_base_path, str(wid))
    os.makedirs(writer_dir, exist_ok=True)

    for i, img_tensor in enumerate(style_list):
        img_np = img_tensor[0]
        # show_img(img_np)
        img_np = img_np.squeeze().numpy()  # Convert to H x W uint8
        # print(img_np)
        img_uint8 = ((img_np + 1.0) * 127.5).clip(0, 255).astype('uint8')


        save_path = os.path.join(writer_dir, f'{i}.png')  # or '.jpg' if preferred
        cv2.imwrite(save_path, img_uint8)
