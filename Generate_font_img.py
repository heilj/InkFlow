from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pickle
import torch
import torch.nn.functional as F

letters = '` ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\'-"/,.+_!#&():;?'
# font = ImageFont.truetype("unifont.ttf", 16)  # or other monospaced font

# symbols = []
# for c in letters:
#     img = Image.new('L', (16, 16), 0)  # white background
#     draw = ImageDraw.Draw(img)
#     draw.text((0, 0), c, font=font, fill=255)  # black character
#     mat = np.array(img) / 255.0
#     symbols.append({'idx': [ord(c)], 'mat': mat.astype(np.float32)})

# # Save
# with open("data/unifont.pickle", "wb") as f:
#     pickle.dump(symbols, f)

# import cv2
# import torch
# import numpy as np
# import torch.nn.functional as F

# lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

# def compute_laplacian(tensor_img):  # shape: (1, H, W)
#     return F.conv2d(tensor_img.unsqueeze(0), lap_kernel, padding=1).squeeze(0)

# def get_symbols(input_type):
#         with open(f"data/{input_type}.pickle", "rb") as f:
#             symbols = pickle.load(f)

#         symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
#         contents = []
#         for char in letters:
#             symbol = torch.from_numpy(symbols[ord(char)]).float()
#             contents.append(symbol)
#         contents.append(torch.zeros_like(contents[0])) # blank image as PAD_TOKEN
#         contents = torch.stack(contents)
#         return contents
# contents = get_symbols('unifont')
# print(contents.shape)

import h5py
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

from lib.alphabet import strLabelConverter
from torchvision import transforms


# === Laplacian generator ===
laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().view(1, 1, 3, 3)
laplacian_kernel.requires_grad = False

def get_laplacian_ref(style_img):
    tensor = torch.from_numpy(style_img).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    laplace = F.conv2d(tensor, laplacian_kernel, padding=1)
    return laplace.squeeze(0)  # (1, H, W)

# === Transform for grayscale image ===
img_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (1, H, W), [0,1]
])

# === Copy and extend h5 file ===
old_path = "data/iam/trnvalset_words64_OrgSz.hdf5"
new_path = "data/iam/trnvalset_words64_OrgSz_lap.hdf5"

with h5py.File(old_path, "r") as f_old, h5py.File(new_path, "w") as f_new:
    # === Copy over existing datasets ===
    for key in f_old.keys():
        f_new.create_dataset(key, data=f_old[key])

    # === Load metadata ===
    imgs = f_old["imgs"][:]
    
    
    lbs = f_old["lbs"][:]
    img_seek_idxs = f_old["img_seek_idxs"][:]
    print(img_seek_idxs.shape)
    img_lens = f_old["img_lens"][:]
    print(type(img_lens))
    lb_seek_idxs = f_old["lb_seek_idxs"][:]
    lb_lens = f_old["lb_lens"][:]

    # === Initialize storage for laplacian and content ===
    num_samples = len(img_lens)


    print("Generating features...")
    laplace_chunks = []
    laplace_seek_idxs = []
    laplace_lens = []

    offset = 0
    for i in tqdm(range(num_samples)):
        seek = img_seek_idxs[i]
        w = img_lens[i]
        img = imgs[:, seek : seek + w]

        img_pil = Image.fromarray(img)
        style_img = np.array(img_pil.convert("L"), dtype=np.float32) / 255.0

        laplace = get_laplacian_ref(style_img)  # (1, H, W)
        lap = laplace[0]  # remove channel dim for consistent storage (H, W)

        laplace_chunks.append(lap)
        laplace_seek_idxs.append(offset)
        laplace_lens.append(lap.shape[1])
        offset += lap.shape[1]

    # Concatenate all laplace images into one big (H, total_width) array
    laplace_concat = np.concatenate(laplace_chunks, axis=1).astype(np.float32)
    laplace_seek_idxs = np.array(laplace_seek_idxs)
    laplace_lens = np.array(laplace_lens)
    print(laplace_concat.shape)
    print(laplace_seek_idxs.shape)
    print(laplace_lens.shape)

    # Store into hdf5
    f_new.create_dataset("laplacian_refs", data=laplace_concat)
    
    f_new.create_dataset("laplacian_seek_idxs", data=np.array(laplace_seek_idxs, dtype=np.int32))
    f_new.create_dataset("laplacian_lens", data=np.array(laplace_lens, dtype=np.int32))



    # f_new.create_dataset("laplacian_refs", data=lap_arr)
