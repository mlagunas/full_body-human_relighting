import glob

from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torchvision import transforms

from transforms import to_tensor
from utils import load_image


def resize_if(img, size):
    c, h, w = img.shape

    if h < size:
        return img
    else:
        # get resulting image size
        reduction = h / size
        res_w = int(w / reduction)
        # set to nearest (mode=0) or bicubic (mode=2) if we have a mask or an image
        mode = 0 if c == 1 else 2
        return transforms.Resize(size=(size, res_w), interpolation=mode)(img)


def _default_trf():
    return transforms.Compose([
        to_tensor,
        lambda x: resize_if(x, 2048),
    ])


class PhotosDataset(Dataset):
    def __init__(self, root, transforms=_default_trf()):

        self.original_paths = list(glob.glob(root + '/original/*'))
        self.mask_paths = list(glob.glob(root + '/mask/*'))
        self.original_paths.sort()
        self.mask_paths.sort()

        self.transforms = transforms

    def __getitem__(self, ix):
        img = load_image(self.original_paths[ix])
        mask = load_image(self.mask_paths[ix])
        name = self.mask_paths[ix].split('/')[-1].split('.')[0]

        if len(mask.shape) == 3:
            mask = mask.sum(axis=-1, keepdims=True) / 3

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        # if the input does not have even dimensions we add one row (or column) of zeros
        pad_h = [0, 0] if img.shape[1] % 2 == 0 else [1, 0]
        pad_w = [0, 0] if img.shape[2] % 2 == 0 else [1, 0]

        img = F.pad(img, pad_w + pad_h)
        mask = F.pad(mask, pad_w + pad_h)

        # return image name and the image and mask (w/ format CxWxH)
        return {
            'name': name,
            'img_pt': img,
            'img_sh': img,
            'mask': mask,
        }

    def __len__(self):
        return len(self.original_paths)
