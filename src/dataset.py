from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("L")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label