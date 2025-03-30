from lib.datasets import Hdf5Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
import random
import torch

dataset = Hdf5Dataset(
    root='data/iam',
    split='trnvalset_words64_OrgSz.hdf5',
    # transforms=Compose([ToTensor(), Normalize([0.5], [0.5])]),
    alphabet_key='iam_word'
)

total_len = len(dataset)
train_len = int(0.9 * total_len)
val_len = total_len - train_len

train_set, val_set = random_split(dataset, [train_len, val_len])


sample = dataset[1]
print(sample.keys())
print(sample['style_img'].shape)
print(sample['org_img'].shape)
print(sample['lb'])
print(sample['wid'])

def show_img(img):
    image_tensor = img
    image = image_tensor.squeeze().numpy()  # shape: [H, W]
    image = image * 0.5 + 0.5  # reverse normalization ([-1,1] → [0,1])

    # Show image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
# show_img(sample['org_img'])


loader = DataLoader(
    val_set,
    batch_size=1,
    shuffle=True,
    collate_fn=Hdf5Dataset.collect_fn
)

for batch in loader:
    # print(batch['org_imgs'].shape)   # [B, 1, H, max_W]
    print(batch.keys()) 
    print(batch['lb_lens'])
    print(batch['style_img_lens'])
    
# num_samples = 8
# k = 2 

# wid_to_indices = {}
# for i in range(len(dataset)):
#     wid = int(dataset.wids[i])
#     wid_to_indices.setdefault(wid, []).append(i)

# selected_wids = random.sample(list(wid_to_indices.keys()), min(5, len(wid_to_indices)))
# # fig, axes = plt.subplots(len(selected_wids), num_samples + k, figsize=(num_samples + k, 5))

# for row, wid in enumerate(selected_wids):
#     indices = random.sample(wid_to_indices[wid], k=k)
#     # ref_imgs = torch.stack([dataset[i]['style_img'] for i in indices])
#     for i in indices:
#         print(dataset[i]['style_img'].shape)
#         show_img(dataset[i]['style_img'])

