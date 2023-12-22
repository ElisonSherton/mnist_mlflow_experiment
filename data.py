import os, random
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class DigitsDataset(Dataset):
    def __init__(self, data_dir, pad_position = "tl", transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_list = self._get_image_list()
        self.pad_position = pad_position

    def _get_image_list(self):
        image_list = [x for x in self.data_dir.glob("**/*") if x.is_file()]
        return image_list

    def _get_class_from_filename(self, file_path):
        return int(file_path.parent.name)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.get_padded(image, self.pad_position)
        
        if self.transform:
            image = self.transform(image)

        class_label = self._get_class_from_filename(img_path)

        return image, class_label

    def get_padded(self, img, type = None):
        # N = 32 # Hindi
        N = 28 # English
        z = np.zeros((2 * N, 2 * N, 3), dtype = np.uint8)
        i = np.array(img)
        if type == "tl":
            z[:N, :N, :] = i
        elif type == "tr":
            z[:N, N:, :] = i
        elif type == "bl":
            z[N:, :N, :] = i
        elif type == "br":
            z[N:, N:, :] = i
        else:
            z[N//2:3 * N // 2, N // 2: 3 * N // 2, :] = i
        return Image.fromarray(z)

    def visualize(self, N = 12):
        imgs_to_plot = random.sample(self.image_list, N)
        fig, ax = plt.subplots(2, 6, figsize = (8,4))
        for i, a in zip(imgs_to_plot, ax.flat):
            img = Image.open(i).convert('RGB')
            img = self.get_padded(img, self.pad_position)
            a.imshow(img); a.set_title(i.parent.name)
            a.set_xticks([]); a.set_yticks([]);
        return fig

    def visualize_at_idx(self, idx):
        img_to_plot = self.image_list[idx]
        fig, a = plt.subplots(1, 1, figsize = (8,4))
        img = Image.open(img_to_plot).convert('RGB')
        img = self.get_padded(img, self.pad_position)
        a.imshow(img); a.set_title(img_to_plot.parent.name)
        a.set_xticks([]); a.set_yticks([]);
        
        
