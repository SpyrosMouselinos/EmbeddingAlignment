import ast
import logging
import os
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.parallel
from fromage import utils
from fromage.utils import get_image_from_url
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from datasets import load_dataset
import random

STOP_WORDS = set(stopwords.words('english'))
PUNC_TABLE = str.maketrans('', '', string.punctuation)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, tokenizer, max_items: int = -1) -> Dataset:
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
                       'caption', args.visual_model, train=train, max_len=args.max_len,
                       image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx,
                       save_images=ast.literal_eval(args.save_images), max_items=max_items)
            for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
    elif len(dataset_paths) == 1:
        dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'image',
                             'caption', args.visual_model, train=train, max_len=args.max_len,
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
                 image_size: int = 224,
                 retrieval_token_idx: int = -1, save_images: bool = False, max_items=-1):
        logging.debug(f'Loading tsv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep, header=None, names=[caption_key, img_key])
        df['img_index'] = list(range(0, df.shape[0]))
        self.max_img_index = df['img_index'].max()
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
        self.retrieval_token_idx = retrieval_token_idx
        self.save_images = save_images
        self.font = None
        self.cache_list_ = []
        logging.debug('Done loading data.')
        self.build_dataset_()

    def build_dataset_(self):
        FILE_PATH = os.path.join(self.base_image_dir, 'corrupted_indexes.txt')
        if os.path.exists(FILE_PATH):
            with open(FILE_PATH, 'r') as fin:
                self.cache_list_ = [int(f) for f in list(set(fin.readline().strip().split(',')))]
        if self.max_items == -1:
            MAX_INDEX = self.max_img_index
        else:
            MAX_INDEX = self.max_items

        for i in range(MAX_INDEX):
            if i in self.cache_list_:
                # print(f"[OLD] Image {i} not found")
                continue

            # ELse

            image_path = os.path.join(self.base_image_dir, str(self.images_index[i]) + '.png')
            if os.path.exists(image_path):
                pass
            else:
                try:
                    img = get_image_from_url(self.images[i])
                    if self.save_images:
                        img.save(image_path)
                except:
                    self.cache_list_.append(i)
                    print(f"[NEW] Image {i} not found")
        with open(FILE_PATH, 'w') as fout:
            lll = len(self.cache_list_)
            for i, j in enumerate(list(set(self.cache_list_))):
                fout.write(str(j))
                if i < lll - 1:
                    fout.write(',')

    def clean_caption(self, caption):
        tokens = word_tokenize(caption)
        tokens = [w.lower() for w in tokens]
        stripped = [w.translate(PUNC_TABLE) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        words = [w for w in words if not w in STOP_WORDS]
        new_caption = ' '.join(words)
        new_caption = new_caption.strip()
        return new_caption

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        while True:
            if idx in self.cache_list_:
                idx = idx + 1
                if idx >= len(self.captions):
                    idx = 0
                continue
            image_path = os.path.join(self.base_image_dir, str(self.images_index[idx]) + '.png')
            caption = str(self.captions[idx])
            # print(f"Caption Before:{caption}\n")
            caption = self.clean_caption(caption)
            # print(f"Caption After:{caption}\n")
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                else:
                    try:
                        img = get_image_from_url(self.images[idx])
                    except:
                        print(f"Image {idx} not found")
                        raise Exception
                    if self.save_images:
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
                return images, tokens, caption_len, idx

            except Exception as e:
                self.cache_list_.append(idx)
                idx = np.random.randint(0, len(self) - 1)


def isvowel(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']
    cap_vowels = ['A', 'E', 'I', 'O', 'u']
    if letter in vowels or letter in cap_vowels:
        return True
    else:
        return False


class ImgDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 feature_extractor_model: str,
                 train: bool = True,
                 max_len: int = 24,
                 image_size: int = 224,
                 return_og_image=False):
        self.max_img_index = len(dataset) - 1
        self.max_items = len(dataset)
        self.images = dataset['img']
        self.fine_captions = dataset['fine_label']

        self.training_mode = 'training' if train else 'validation'

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor = utils.get_feature_extractor_for_model(
            feature_extractor_model, image_size=image_size, train=False)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fine_label_translation_table = {
            0: 'apple',

            1: 'aquarium fish',

            2: 'baby',

            3: 'bear',

            4: 'beaver',

            5: 'bed',

            6: 'bee',

            7: 'beetle',

            8: 'bicycle',

            9: 'bottle',

            10: 'bowl',

            11: 'boy',

            12: 'bridge',

            13: 'bus',

            14: 'butterfly',

            15: 'camel',

            16: 'can',

            17: 'castle',

            18: 'caterpillar',

            19: 'cattle',

            20: 'chair',

            21: 'chimpanzee',

            22: 'clock',

            23: 'cloud',

            24: 'cockroach',

            25: 'couch',

            26: 'cra',

            27: 'crocodile',

            28: 'cup',

            29: 'dinosaur',

            30: 'dolphin',

            31: 'elephant',

            32: 'flatfish',

            33: 'forest',

            34: 'fox',

            35: 'girl',

            36: 'hamster',

            37: 'house',

            38: 'kangaroo',

            39: 'keyboard',

            40: 'lamp',

            41: 'lawn mower',

            42: 'leopard',

            43: 'lion',

            44: 'lizard',

            45: 'lobster',

            46: 'man',

            47: 'maple tree',

            48: 'motorcycle',

            49: 'mountain',

            50: 'mouse',

            51: 'mushroom',

            52: 'oak tree',

            53: 'orange',

            54: 'orchid',

            55: 'otter',

            56: 'palm tree',

            57: 'pear',

            58: 'pickup truck',

            59: 'pine tree',

            60: 'plain',

            61: 'plate',

            62: 'poppy',

            63: 'porcupine',

            64: 'possum',

            65: 'rabbit',

            66: 'raccoon',

            67: 'ray',

            68: 'road',

            69: 'rocket',

            70: 'rose',

            71: 'sea',

            72: 'seal',

            73: 'shark',

            74: 'shrew',

            75: 'skunk',

            76: 'skyscraper',

            77: 'snail',

            78: 'snake',

            79: 'spider',

            80: 'squirrel',

            81: 'streetcar',

            82: 'sunflower',

            83: 'sweet pepper',

            84: 'table',

            85: 'tank',

            86: 'telephone',

            87: 'television',

            88: 'tiger',

            89: 'tractor',

            90: 'train',

            91: 'trout',

            92: 'tulip',

            93: 'turtle',

            94: 'wardrobe',

            95: 'whale',

            96: 'willow tree',

            97: 'wolf',

            98: 'woman',

            99: 'worm',
        }
        self.build_captions()
        self.return_og_image = return_og_image

    def build_captions(self):
        # self.captions = [' ' + self.fine_label_translation_table[f] for f in self.fine_captions]
        self.no_space_captions = [' ' + self.fine_label_translation_table[f] for f in self.fine_captions]
        self.no_space_captions = ['n ' + f if isvowel(f) else f for f in self.no_space_captions]

    def __len__(self):
        return len(self.no_space_captions)

    def __getitem__(self, idx):
        images = utils.get_pixel_values_for_model(self.feature_extractor, self.images[idx])
        # # Captions #
        # tokenized_data = self.tokenizer(
        #     self.captions[idx],
        #     return_tensors="pt",
        #     padding='max_length',
        #     truncation=True,
        #     max_length=self.max_len)
        # caption_len_1 = tokenized_data.attention_mask[0][1:2].sum()
        # tokens_1 = tokenized_data.input_ids[0][1:2]

        # No space Captions #
        tokenized_data = self.tokenizer(
            self.no_space_captions[idx],
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_len, add_special_tokens=False)
        caption_len_2 = tokenized_data.attention_mask[0].sum()
        tokens_2 = tokenized_data.input_ids[0]

        # tokens = torch.cat([tokens_1, tokens_2], dim=0)
        tokens = tokens_2
        # caption_len = caption_len_1 + caption_len_2
        caption_len = caption_len_2
        if self.return_og_image:
            return torchvision.transforms.PILToTensor()(self.images[idx]), images, tokens, caption_len, idx
        else:
            return images, tokens, caption_len, idx


def get_cifar100_dataset(args, split: str, tokenizer, return_og_image=False) -> Dataset:
    assert split in ['train', 'val'
                     ], 'Expected split to be one of "train" or "val", got {split} instead.'
    train = split == 'train'

    if split == 'train':
        dataset = load_dataset("cifar100", split="train")
        dataset = dataset.train_test_split(test_size=0.5, stratify_by_column="fine_label")['train']

    elif split == 'val':
        dataset = load_dataset("cifar100", split="test")[:5000]

    xxx = ImgDataset(dataset=dataset, tokenizer=tokenizer, feature_extractor_model=args.visual_model, train=train,
                     max_len=args.max_len,
                     image_size=args.image_size, return_og_image=return_og_image)
    return xxx
