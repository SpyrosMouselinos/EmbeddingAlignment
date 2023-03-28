import ast
import random
from typing import Optional, Tuple
import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from fromage import utils
from fromage.utils import get_image_from_url


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, tokenizer, max_items: int=-1) -> Dataset:
    assert split in ['train', 'val'
                     ], 'Expected split to be one of "train" or "val", got {split} instead.'

    dataset_paths = []
    image_data_dirs = []
    train = split == 'train'

    # Default configs for datasets.
    # Folder structure should look like:
    if split == 'train':
        if 'cc3m' in args.dataset:
            dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train.tsv'))
            image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/training/'))
        else:
            raise NotImplementedError

    elif split == 'val':
        if 'cc3m' in args.val_dataset:
            dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_val.tsv'))
            image_data_dirs.append(os.path.join(args.image_dir, 'cc3m/validation'))
        else:
            raise NotImplementedError

        assert len(dataset_paths) == len(image_data_dirs) == 1, (dataset_paths, image_data_dirs)
    else:
        raise NotImplementedError

    if len(dataset_paths) > 1:
        print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
        dataset = torch.utils.data.ConcatDataset([
            CsvDataset(path, image_dir, tokenizer, 'image',
                       'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
                       image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx,
                       save_images=ast.literal_eval(args.save_images), max_items=max_items)
            for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
    elif len(dataset_paths) == 1:
        dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'image',
                             'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
                             image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx,
                             save_images=ast.literal_eval(args.save_images), max_items=max_items)
    else:
        raise ValueError(
            f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
    return dataset


class CsvDataset(Dataset):
    def __init__(self,
                 input_filename,
                 base_image_dir,
                 tokenizer,
                 img_key,
                 caption_key,
                 feature_extractor_model: str,
                 train: bool = True,
                 max_len: int = 32,
                 sep="\t",
                 precision: str = 'fp16',
                 image_size: int = 224,
                 retrieval_token_idx: int = -1, save_images: bool = False, max_items=-1):
        logging.debug(f'Loading tsv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep, header=None, names=[caption_key, img_key])
        df['img_index'] = list(range(0, df.shape[0]))
        self.max_items = max_items
        self.base_image_dir = base_image_dir
        self.images = df[img_key].tolist()[:self.max_items]
        self.captions = df[caption_key].tolist()[:self.max_items]
        self.images_index = df['img_index'].tolist()[:self.max_items]
        self.training_mode = 'training' if train else 'validation'
        assert len(self.images) == len(self.captions)

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor = utils.get_feature_extractor_for_model(
            feature_extractor_model, image_size=image_size, train=False)
        self.image_size = image_size

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.precision = precision
        self.retrieval_token_idx = retrieval_token_idx
        self.save_images = save_images
        self.font = None


        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        while True:

            image_path = os.path.join(self.base_image_dir, str(self.images_index[idx]) + '.png')
            caption = str(self.captions[idx])

            try:
                try:
                    img = Image.open(image_path)
                except FileNotFoundError:
                    print(f"{self.training_mode} image no {self.images_index[idx]} not found... downloading...\n")
                    img = get_image_from_url(self.images[idx])
                    if self.save_images:
                        print(f"Saving at: {image_path}")
                        img.save(image_path)

                images = utils.get_pixel_values_for_model(self.feature_extractor, img)

                tokenized_data = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len)
                tokens = tokenized_data.input_ids[0]
                caption_len = tokenized_data.attention_mask[0].sum()
                return image_path, images, tokens, caption_len

            except Exception as e:
                print(f'Error reading {image_path} with caption {caption}: {e}')
                # Pick a new example at random.
                idx = np.random.randint(0, len(self) - 1)