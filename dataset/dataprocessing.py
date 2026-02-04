import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, height, width, augment=False):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.height = height
        self.width = width
        self.augment = augment

        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        self.image_paths = [os.path.join(image_dir, f) for f in image_files]
        self.mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.augment:
            transform = transforms.Compose([
                transforms.Resize((int(self.height * 1.25), int(self.width * 1.25))),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(16, fill=144),
                transforms.CenterCrop((self.height, self.width)),
                transforms.ToTensor()
            ])

            seed = np.random.randint(42)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            image = transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            mask = transform(mask)
        else:
            transform = transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor()
            ])
            image = transform(image)
            mask = transform(mask)

        image = self.normalize(image)
        mask[mask > 0] = 1.0

        return {
            'image': image,
            'mask': mask,
            'A_paths': self.image_paths[idx]
        }
