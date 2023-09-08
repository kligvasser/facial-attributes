import torch
import numpy as np
import os

from PIL import Image
from torchvision import transforms

import utils.misc
import data.misc


POS_VALUE = 1
NEG_VALUE = 0
ID_TO_CLS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]
CLS_TO_ID = {v: k for k, v in enumerate(ID_TO_CLS)}


class ImagesLibary(torch.utils.data.Dataset):
    def __init__(self, root='', image_size=None, max_size=None):
        self.root = root
        self.image_size = image_size
        self.max_size = max_size

        self._init()

    def _init(self):
        self.paths = data.misc.get_path_list(self.root, self.max_size)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def preprocess_image(self, path):
        image = Image.open(path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.preprocess(image)
        return image

    def __getitem__(self, index):
        example = dict()

        path = self.paths[index]
        example['image'] = self.preprocess_image(path)
        example['path'] = path

        return example

    def __len__(self):
        return len(self.paths)


class BalancedSampling(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        pkl_path='',
        root='',
        training=True,
        max_size=None,
    ):
        self.pkl_path = pkl_path
        self.root = root
        self.training = training

        self.df = utils.misc.load_df(pkl_path)

        if max_size:
            self.df = self.df.head(max_size)

        self.cls_list = list(self.df.columns.values)
        self.num_cls = len(self.cls_list)

        self._set_transform()

    def _set_transform(self):
        if self.training:
            self.transform = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)
                    ),
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomGrayscale(0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __getitem__(self, index):
        cls_id = index % self.num_cls
        cls = self.cls_list[cls_id]

        base_name = np.random.choice(list((self.df[self.df[cls] == POS_VALUE]).index))
        image_path = os.path.join(self.root, base_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        row = self.df.loc[base_name]
        label = row.values.astype(float)

        return {'input': image, 'label': label, 'class': cls}

    def __len__(self):
        return len(self.df)
