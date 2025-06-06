import os, h5py
import numpy as np
from PIL import Image
import cv2
from copy import deepcopy
import itertools
import glob

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from lib.alphabet import strLabelConverter, Alphabets
from lib.path_config import data_roots, data_paths, ImgHeight, CharWidth
from lib.transforms import RandomScale, RandomClip
import pickle
import random
import math

from torch.utils.data import Sampler
from collections import defaultdict

class LengthAwareSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.lengths = [dataset.img_lens[i] for i in indices]  # precompute img_lens
        # print(self.img_lens)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets=32, shuffle_buckets=True, shuffle_within_bucket=True, 
                 num_replicas=1, rank=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle_buckets = shuffle_buckets
        self.shuffle_within_bucket = shuffle_within_bucket
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.lengths = dataset.lengths
        self.indices = list(range(len(dataset)))
        
        self._batches = None
        
    def _create_batches(self):
        # Create a list of (idx, length) pairs and sort by length
        idx_lengths = list(zip(self.indices, self.lengths))
        sorted_idx_lengths = sorted(idx_lengths, key=lambda x: x[1])
        
        # Divide indices into buckets of similar lengths
        buckets = []
        samples_per_bucket = len(sorted_idx_lengths) // self.num_buckets
        for i in range(0, len(sorted_idx_lengths), samples_per_bucket):
            if i + samples_per_bucket <= len(sorted_idx_lengths):
                buckets.append(sorted_idx_lengths[i:i+samples_per_bucket])
            elif not self.drop_last:
                buckets[-1].extend(sorted_idx_lengths[i:])
        
        # Shuffle within each bucket if required
        if self.shuffle_within_bucket:
            for bucket in buckets:
                random.shuffle(bucket)
        
        # Extract indices from buckets
        bucket_indices = [[idx for idx, _ in bucket] for bucket in buckets]
        
        # Create batches from each bucket
        batches = []
        for indices in bucket_indices:
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    batches.append(batch)
        
        # Shuffle the batches if required
        if self.shuffle_buckets:
            random.shuffle(batches)
        
        # DDP slicing
        batches = batches[self.rank::self.num_replicas]
        
        return batches
    
    def __iter__(self):
        self._batches = self._create_batches()
        return iter(self._batches)
    
    def __len__(self):
        if self._batches is None:
            self._batches = self._create_batches()
        return len(self._batches)
    
    def set_epoch(self, epoch):
        # For reshuffling across epochs
        self.epoch = epoch
        random.seed(epoch)  # This ensures different shuffling per epoch







class Hdf5Dataset(Dataset):
    def __init__(self, root, split, transforms=None, alphabet_key='all', process_style=False, normalize_wid=True):
        super(Hdf5Dataset, self).__init__()
        self.root = root
        self._load_h5py(os.path.join(self.root, split), normalize_wid)
        self.transforms = transforms
        self.org_transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
        self.label_converter = strLabelConverter(alphabet_key)
        self.letters = Alphabets['all']
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.process_style = process_style
        
        self.con_symbols = self.get_symbols('unifont')
        

    def _load_h5py(self, file_path, normalize_wid=True):
        # print(self.file_path)
        self.file_path = file_path
        if os.path.exists(self.file_path):
            h5f = h5py.File(self.file_path, 'r')
            self.imgs, self.lbs = h5f['imgs'][:], h5f['lbs'][:]
            # self.laplace_refs = h5f['laplacian_refs'][:]
            self.img_seek_idxs, self.lb_seek_idxs = h5f['img_seek_idxs'][:], h5f['lb_seek_idxs'][:]
            # self.laplace_seek_idxs = h5f['laplacian_seek_idxs'][:]
            self.img_lens, self.lb_lens = h5f['img_lens'][:], h5f['lb_lens'][:]
            # self.laplace_lens = h5f['laplacian_lens'][:]
            self.wids = h5f['wids'][:]
            if normalize_wid:
                self.wids -= self.wids.min()
            # print(self.lb_lens)
            valid_idxs = (self.img_lens <= 352)

            self.img_seek_idxs = self.img_seek_idxs[valid_idxs]
            self.lb_seek_idxs = self.lb_seek_idxs[valid_idxs]
            self.img_lens = self.img_lens[valid_idxs]
            self.lb_lens = self.lb_lens[valid_idxs]
            self.wids = self.wids[valid_idxs]
            h5f.close()
            
        else:
            print(self.file_path, ' does not exist!')
            self.imgs, self.lbs = None, None
            self.img_seek_idxs, self.lb_seek_idxs =  None, None
            self.img_lens, self.lb_lens =  None, None
            self.wids = None

    def get_symbols(self, input_type):
        with open(f"data/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0])) # blank image as PAD_TOKEN
        contents = torch.stack(contents)
        return contents

    def __getitem__(self, idx):
        data = {}
        img_seek_idx, img_len = self.img_seek_idxs[idx], self.img_lens[idx]
        lb_seek_idx, lb_len = self.lb_seek_idxs[idx], self.lb_lens[idx]
        # laplace_seek_idx, laplace_len = self.laplace_seek_idxs[idx], self.laplace_lens[idx]
        img = self.imgs[:, img_seek_idx : img_seek_idx + img_len]
        text = ''.join(chr(ch) for ch in self.lbs[lb_seek_idx : lb_seek_idx + lb_len])
        # laplace_ref = self.laplace_refs[:, laplace_seek_idx : laplace_seek_idx + laplace_len]
        data['text'] = text
        lb = self.label_converter.encode(text)
        content = [self.letter2index[i] for i in text]
        content = self.con_symbols[content]
        data['content'] = content
        wid = self.wids[idx]
        data['lb'], data['wid'] = lb, wid
        data['org_img'] = self.org_transforms(Image.fromarray(deepcopy(img), mode='L'))
        # data['laplace_img'] = self.org_transforms(Image.fromarray(deepcopy(laplace_ref), mode='L'))


        # style image
        if self.process_style:
            h, w = img.shape[:2]
            new_w = CharWidth * len(text)
            dim = (new_w, ImgHeight)
            if new_w < w:
                style_img = cv2.resize(deepcopy(img), dim, interpolation=cv2.INTER_AREA)
            else:
                style_img = cv2.resize(deepcopy(img), dim, interpolation=cv2.INTER_LINEAR)
            style_img = Image.fromarray(style_img, mode='L')

        else:
            style_img = Image.fromarray(deepcopy(img), mode='L')

        data['style_img'] = self.org_transforms(deepcopy(style_img))

        if self.transforms is not None:
            data['aug_img'] = self.transforms(style_img)

        return data

    def __len__(self):
        return len(self.img_lens)

    @staticmethod
    def _recalc_len(leng, scale=CharWidth):
        tmp = leng % scale
        return leng + scale - tmp if tmp != 0 else leng

    def _pad_to_multiple(length, multiple=8):
        return (length + multiple - 1) // multiple * multiple
    
    def get_style_ref(wr_id):
        wr_id = str(wr_id)
        style_path = 'data/style_wids'
        style_list = os.listdir(os.path.join(style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2) # anchor and positive
        style_images = [cv2.imread(os.path.join(style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        # laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
        #                   for index in style_index]
        height =64
        # height = style_images[0].shape[0]
        # assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])
        
        '''style images'''
        style_images = [style_image / 127.5 - 1.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        # laplace_images = [laplace_image/255.0 for laplace_image in laplace_images]
        # new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        # new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        # new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        # return new_style_images, new_laplace_images
        return new_style_images

    @staticmethod
    def vanilla_collect_fn(batch):
        org_imgs, org_img_lens, style_imgs, style_img_lens, aug_imgs, aug_img_lens,\
        lbs, lb_lens, wids = [], [], [], [], [], [], [], [], []

        for data in batch:
            org_img, style_img, lb, wid = data['org_img'], data['style_img'], data['lb'], data['wid']
            aug_img = data['aug_img'] if 'aug_img' in data else None
            if isinstance(org_img, torch.Tensor):
                org_img = org_img.numpy()
            if isinstance(style_img, torch.Tensor):
                style_img = style_img.numpy()
            if aug_img is not None and isinstance(aug_img, torch.Tensor):
                aug_img = aug_img.numpy()

            org_imgs.append(org_img)
            org_img_lens.append(org_img.shape[-1])
            style_imgs.append(style_img)
            style_img_lens.append(style_img.shape[-1])
            lbs.append(lb)
            lb_lens.append(len(lb))
            wids.append(wid)
            if aug_img is not None:
                aug_imgs.append(aug_img)
                aug_img_lens.append(Hdf5Dataset._recalc_len(aug_img.shape[-1]))

        bdata = {}
        bz = len(lb_lens)
        pad_org_img_max_len = Hdf5Dataset._recalc_len(max(org_img_lens))
        pad_org_imgs = -np.ones((bz, 1, org_imgs[0].shape[-2], pad_org_img_max_len))
        for i, (org_img, org_img_len) in enumerate(zip(org_imgs, org_img_lens)):
            pad_org_imgs[i, 0, :, :org_img_len] = org_img
        bdata['org_imgs'] = torch.from_numpy(pad_org_imgs).float()
        bdata['org_img_lens'] = torch.IntTensor(org_img_lens)

        pad_style_img_max_len = Hdf5Dataset._recalc_len(max(style_img_lens))
        pad_style_imgs = -np.ones((bz, 1, style_imgs[0].shape[-2], pad_style_img_max_len))
        for i, (style_img, style_img_len) in enumerate(zip(style_imgs, style_img_lens)):
            pad_style_imgs[i, 0, :, :style_img_len] = style_img
        bdata['style_imgs'] = torch.from_numpy(pad_style_imgs).float()
        bdata['style_img_lens'] = torch.IntTensor(style_img_lens)

        pad_lbs = np.zeros((bz, max(lb_lens)))
        for i, (lb, lb_len) in enumerate(zip(lbs, lb_lens)):
            pad_lbs[i, :lb_len] = lb
        bdata['lbs'] = torch.from_numpy(pad_lbs).long()
        bdata['lb_lens'] = torch.Tensor(lb_lens).int()
        bdata['wids'] = torch.Tensor(wids).long()

        if len(aug_imgs) > 0:
            pad_aug_imgs = -np.ones((bz, 1, aug_imgs[0].shape[-2], max(aug_img_lens)))
            for i, aug_img in enumerate(aug_imgs):
                pad_aug_imgs[i, 0, :, :aug_img.shape[-1]] = aug_img

            bdata['aug_imgs'] = torch.from_numpy(pad_aug_imgs).float()
            bdata['aug_img_lens'] = torch.IntTensor(aug_img_lens)

        return bdata


    @staticmethod
    def collect_fn(batch, pad_multiple=16):
        org_imgs, org_img_lens, style_imgs, style_img_lens, aug_imgs, aug_img_lens,\
        lbs, lb_lens, wids, laplacian_imgs, laplacian_refs, style_refs = [], [], [], [], [], [], [], [], [], [], [], []
        c_width = [len(item['content']) for item in batch]
        content_ref = torch.zeros([len(batch), max(c_width), 16 , 16], dtype=torch.float32)
        for idx, data in enumerate(batch):
            org_img, lb, wid = data['org_img'], data['lb'], data['wid']
            # laplacian_img = data['laplace_img']
            aug_img = data['aug_img'] if 'aug_img' in data else None
            style_img = Hdf5Dataset.get_style_ref(wid)
            if isinstance(org_img, torch.Tensor): org_img = org_img.numpy()
            if isinstance(style_img, torch.Tensor): style_img = style_img.numpy()
            if aug_img is not None and isinstance(aug_img, torch.Tensor): aug_img = aug_img.numpy()

            org_imgs.append(org_img)
            org_img_lens.append(org_img.shape[-1])
            
            style_imgs.append(style_img)
            # laplacian_imgs.append(laplacian_img)
            style_img_lens.append(style_img.shape[-1])
            lbs.append(lb)
            content = data['content']
            content_ref[idx, :len(content)] = content
            lb_lens.append(len(lb))
            wids.append(wid)
            if aug_img is not None:
                aug_imgs.append(aug_img)
                aug_img_lens.append(Hdf5Dataset._recalc_len(aug_img.shape[-1]))

        bdata = {}
        bz = len(lb_lens)

        # === Pad original images ===
        pad_org_img_max_len = max(org_img_lens)
        pad_org_img_max_len = Hdf5Dataset._pad_to_multiple(pad_org_img_max_len, multiple=pad_multiple)
        pad_org_imgs = -np.ones((bz, 1, org_imgs[0].shape[-2], pad_org_img_max_len))
        for i, (org_img, org_img_len) in enumerate(zip(org_imgs, org_img_lens)):
            pad_org_imgs[i, 0, :, :org_img_len] = org_img
        # pad_laplacian_imgs = -np.ones((bz, 1, laplacian_imgs[0].shape[-2], pad_org_img_max_len))
        # for i, (laplacian_img, org_img_len) in enumerate(zip(laplacian_imgs, org_img_lens)):
            # pad_laplacian_imgs[i, 0, :, :org_img_len] = laplacian_img
        bdata['org_imgs'] = torch.from_numpy(pad_org_imgs).float()
        # bdata['laplacian_imgs'] = torch.from_numpy(pad_laplacian_imgs).float()
        bdata['org_img_lens'] = torch.IntTensor(org_img_lens)

        # === Pad style images ===
        pad_style_img_max_len = max(style_img_lens)
        pad_style_img_max_len = Hdf5Dataset._pad_to_multiple(pad_style_img_max_len, multiple=pad_multiple)
        pad_style_imgs = -np.ones((bz, 2, style_imgs[0].shape[-2], pad_style_img_max_len))
        for i, (style_img, style_img_len) in enumerate(zip(style_imgs, style_img_lens)):
            pad_style_imgs[i, :, :, :style_img_len] = style_img
        bdata['style_imgs'] = torch.from_numpy(pad_style_imgs).float()
        bdata['style_img_lens'] = torch.IntTensor(style_img_lens)

        # === Pad labels ===
        pad_lbs = np.zeros((bz, max(lb_lens)))
        for i, (lb, lb_len) in enumerate(zip(lbs, lb_lens)):
            pad_lbs[i, :lb_len] = lb
        bdata['lbs'] = torch.from_numpy(pad_lbs).long()
        bdata['lb_lens'] = torch.Tensor(lb_lens).int()
        bdata['wids'] = torch.Tensor(wids).long()

        # === Pad augmented images if any ===
        if len(aug_imgs) > 0:
            pad_aug_img_max_len = max(aug_img_lens)
            pad_aug_img_max_len = Hdf5Dataset._pad_to_multiple(pad_aug_img_max_len, multiple=pad_multiple)
            pad_aug_imgs = -np.ones((bz, 1, aug_imgs[0].shape[-2], pad_aug_img_max_len))
            for i, aug_img in enumerate(aug_imgs):
                pad_aug_imgs[i, 0, :, :aug_img.shape[-1]] = aug_img

            bdata['aug_imgs'] = torch.from_numpy(pad_aug_imgs).float()
            bdata['aug_img_lens'] = torch.IntTensor(aug_img_lens)

        # content_ref = 1.0 - content_ref # invert the image
        bdata['content'] = content_ref

        return bdata

    @staticmethod
    def sort_collect_fn_style(batch):
        batch = Hdf5Dataset.collect_fn(batch)

        style_img_lens = batch['style_img_lens']
        idx = np.argsort(style_img_lens.cpu().numpy())[::-1]

        for key, val in batch.items():
            batch[key] = torch.stack([val[i] for i in idx]).detach()
            print('%15s'%key, batch[key].size(), batch[key].dim())
        return batch

    @staticmethod
    def sort_collect_fn_aug(batch):
        batch = Hdf5Dataset.collect_fn(batch)

        style_img_lens = batch['aug_img_lens']
        idx = np.argsort(style_img_lens.cpu().numpy())[::-1]

        for key, val in batch.items():
            batch[key] = torch.stack([val[i] for i in idx]).detach()
        return batch
    @staticmethod

    def sorted_collect_fn_for_ctc(batch):
        batch = Hdf5Dataset.collect_fn(batch)

        # Use the lengths corresponding to the recognizer input (likely org_img_lens)
        sort_idx = batch['org_img_lens'].argsort(descending=True)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.size(0) == sort_idx.size(0):
                batch[k] = v[sort_idx]
        return batch



    @staticmethod
    def merge_batch(batch1, batch2, device):
        lbs1, lb_lens1, wids1 = batch1['lbs'], batch1['lb_lens'], batch1['wids']
        lbs2, lb_lens2, wids2 = batch2['lbs'], batch2['lb_lens'], batch2['wids']
        bz1, bz2 = lb_lens1.size(0), lb_lens2.size(0)

        mbdata = {}
        for img_key, img_len_key in [('org_imgs', 'org_img_lens'),
                                     ('style_imgs', 'style_img_lens'),
                                     ('aug_imgs', 'aug_img_lens')]:
            if img_len_key not in batch1: continue

            imgs1, imgs2 =  batch1[img_key], batch2[img_key]
            img_lens1, img_lens2 = batch1[img_len_key], batch2[img_len_key]
            max_img_len = max(imgs1.size(-1), imgs2.size(-1))
            pad_imgs = -torch.ones((bz1 + bz2, imgs1.size(1), imgs1.size(2), max_img_len)).float().to(device)
            pad_imgs[:bz1, :, :, :imgs1.size(-1)] = imgs1
            pad_imgs[bz1:, :, :, :imgs2.size(-1)] = imgs2
            merge_img_lens = torch.cat([img_lens1, img_lens2]).to(device)

            mbdata[img_key] = pad_imgs
            mbdata[img_len_key] = merge_img_lens

        max_lb_len = max(lb_lens1.max(), lb_lens2.max()).item()
        pad_lbs = torch.zeros((bz1 + bz2, max_lb_len)).long().to(device)
        pad_lbs[:bz1, :lbs1.size(-1)] = lbs1
        pad_lbs[bz1:, :lbs2.size(-1)] = lbs2
        mbdata['lbs'] = pad_lbs
        merge_lb_lens = torch.cat([lb_lens1, lb_lens2]).to(device)
        mbdata['lb_lens'] = merge_lb_lens
        merge_wids = torch.cat([wids1, wids2]).long().to(device)
        mbdata['wids'] = merge_wids
        return mbdata

    @staticmethod
    def gen_h5file(all_imgs, all_texts, all_wids, save_path):
        img_seek_idxs, img_lens = [], []
        cur_seek_idx = 0
        for img in all_imgs:
            img_seek_idxs.append(cur_seek_idx)
            img_lens.append(img.shape[-1])
            cur_seek_idx += img.shape[-1]

        lb_seek_idxs, lb_lens = [], []
        cur_seek_idx = 0
        for lb in all_texts:
            lb_seek_idxs.append(cur_seek_idx)
            lb_lens.append(len(lb))
            cur_seek_idx += len(lb)

        save_imgs = np.concatenate(all_imgs, axis=-1)
        save_texts = list(itertools.chain(*all_texts))
        save_lbs = [ord(ch) for ch in save_texts]
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('imgs',
                           data=save_imgs,
                           compression='gzip',
                           compression_opts=4,
                           dtype=np.uint8)
        h5f.create_dataset('lbs',
                           data=save_lbs,
                           dtype=np.int32)
        h5f.create_dataset('img_seek_idxs',
                           data=img_seek_idxs,
                           dtype=np.int64)
        h5f.create_dataset('img_lens',
                           data=img_lens,
                           dtype=np.int16)
        h5f.create_dataset('lb_seek_idxs',
                           data=lb_seek_idxs,
                           dtype=np.int64)
        h5f.create_dataset('lb_lens',
                           data=lb_lens,
                           dtype=np.int16)
        h5f.create_dataset('wids',
                           data=all_wids,
                           dtype=np.int16)
        h5f.close()
        print('save->', save_path)


class ImageDataset(Hdf5Dataset):
    ImgHeight = 64

    def __init__(self, *args, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)

    def _load_h5py(self, file_path, normalize_wid=True):
        assert os.path.exists(file_path), file_path + "does not exist!"

        all_imgs, all_writer, all_texts = [], [], []
        fileExtensions = ["jpg", "jpeg", "png", "bmp", "gif"]
        listOfFiles = []
        for extension in fileExtensions:
            listOfFiles.extend(glob.glob(os.path.join(file_path, "*."+ extension)))

        for fn in listOfFiles:
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            img = Image.fromarray(img).convert('L')
            img = np.array(img)

            # Read image labels
            writer_name = os.path.basename(fn).split('.')[0]

            # Normalize image-height
            h, w = img.shape[:2]
            r = self.ImgHeight / float(h)
            new_w = int(w * r)
            new_w = ((new_w + 8) // 16) * 16 
            dim = (new_w, self.ImgHeight)
            # new_w = max(int(w * r), int(ImgHeight / 4 * len(label_text)))
            # dim = (new_w, ImgHeight)
            if new_w < w:
                resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            else:
                resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
            res_img = 255 - resize_img

            all_imgs.append(res_img)
            all_writer.append(writer_name)
            all_texts.append('pseudo')

        '''========prepare image dataset==========='''
        img_seek_idxs, img_lens = [], []
        cur_seek_idx = 0
        for img in all_imgs:
            img_seek_idxs.append(cur_seek_idx)
            img_lens.append(img.shape[-1])
            cur_seek_idx += img.shape[-1]

        lb_seek_idxs, lb_lens = [], []
        cur_seek_idx = 0
        for lb in all_texts:
            lb_seek_idxs.append(cur_seek_idx)
            lb_lens.append(len(lb))
            cur_seek_idx += len(lb)

        self.imgs = np.concatenate(all_imgs, axis=-1).astype(np.uint8)
        save_texts = list(itertools.chain(*all_texts))
        self.lbs = [ord(ch) for ch in save_texts]
        self.img_seek_idxs, self.lb_seek_idxs =\
            np.array(img_seek_idxs).astype(np.int64), np.array(lb_seek_idxs).astype(np.int64)
        # self.img_seek_idxs=\
        #     np.array(img_seek_idxs).astype(np.int64)
        self.img_lens, self.lb_lens = \
            np.array(img_lens).astype(np.int32), np.array(lb_lens).astype(np.int32)
        # self.img_lens= \
        #     np.array(img_lens).astype(np.int32)
        self.writers = all_writer
        self.wids = np.zeros((len(all_imgs),)).astype(np.int32)
        self.org_imgs = self.imgs.copy()  
        self.org_img_lens = self.img_lens.copy()


def get_dataset(dset_name, split, wid_aug=False, recogn_aug=False, process_style=False):
    name = dset_name.strip()
    tag = name.split('_')[0]
    alphabet_key = 'rimes_word' if tag.startswith('rimes') else 'all'

    transforms = [ToTensor(), Normalize([0.5], [0.5])]
    if recogn_aug:
        transforms = [RandomScale()] + transforms
    if wid_aug:
        transforms = [RandomClip()] + transforms
    if not recogn_aug and not wid_aug:
        transforms = None
    else:
        transforms = Compose(transforms)

    if dset_name.startswith('custom'):
        dataset = ImageDataset(root=split, split='',
                               transforms=transforms,
                               alphabet_key=alphabet_key,
                               process_style=process_style)
    else:
        dataset = Hdf5Dataset(data_roots[tag],
                              data_paths[name][split],
                              transforms=transforms,
                              alphabet_key=alphabet_key,
                              process_style=process_style)
    return dataset


def get_collect_fn(sort_input=False, sort_style=True):
    if sort_input:
        if sort_style:
            return Hdf5Dataset.sort_collect_fn_style
        else:
            return Hdf5Dataset.sort_collect_fn_aug
    else:
        return Hdf5Dataset.collect_fn


def get_alphabet_from_corpus(corpus_path):
    items = []
    with open(corpus_path, 'r') as f:
        for line in f.readlines():
            items.append(line.strip())
    alphabet = ''.join(sorted(list(set(''.join(items)))))
    return alphabet


def get_max_image_width(dset):
    max_image_width = 0
    for img, _, _ in dset:
        max_image_width = max(max_image_width, img.size(-1))
    return max_image_width