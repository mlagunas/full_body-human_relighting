import os
import imageio
import numpy as np
import math
import glob
from PIL import Image
from transforms import to_tensor


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_SH_ixs(n_shcoeffs=36):
    def one_ix(ix):
        # get L and M, in spherical harmonics
        l = int(math.sqrt(ix))
        m = ix - l * l - l

        # get row column
        r, c = (l, l + m) if m <= 0 else (l - m, l)

        # get final ix taking into account the bands in SH
        return r * sh_bands + c

    sh_bands = math.ceil(math.sqrt(n_shcoeffs))
    return np.array([one_ix(i) for i in range(n_shcoeffs)])


def load_light_from_dir(dir, ncoeffs=36):
    all_light_files = glob.glob(dir + '/*')
    assert len(all_light_files) > 0, 'No files found in %s' % dir
    all_light_files.sort()
    all_lights = [None] * len(all_light_files)
    for light_file in all_light_files:
        light_rot = light_file.split('iblrot')[-1].split('_')[0]
        all_lights[int(light_rot)] = load_light_coeff(light_file, ncoeffs)
    return all_lights


def load_light_coeff(path, ncoeffs):
    light = load_image(path).reshape(-1, 3)
    sh_ixs = get_SH_ixs(light.shape[0])

    # move from packed to linear representation of SH
    light = light[sh_ixs]

    # load only the required coeffs and move to torch
    return to_tensor(light[:ncoeffs].transpose())


def load_image(img_path):
    ext = os.path.splitext(img_path)[1]
    if ext in ['.exr', '.pfm', '.hdr']:
        return imageio.imread(img_path).copy()
    elif ext in ['.png', '.jpg', '.jpeg']:
        return np.asarray(pil_loader(img_path), np.float32) / 255.
