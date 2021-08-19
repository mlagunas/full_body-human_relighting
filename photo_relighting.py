import os
import imageio
import numpy as np
import torch
import math
from tqdm import tqdm
from pl_module import SigasiaSystem
from dataset import PhotosDataset
from utils import load_light_from_dir
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.utilities.parsing import AttributeDict


def torch_to_np(tensor):
    nptensor = (tensor.clamp(0, 1).cpu().detach().permute(1, 2, 0) * 255).numpy()
    return nptensor.astype(np.uint8)


def np_to_torch(nptensor):
    return torch.tensor(nptensor / 255.).permute(2, 0, 1).unsqueeze(0)


def store_tensor_to_img(tensor, out_name):
    mask = 1 - tensor[-1:] ** 5
    tensor = tensor[:-1]
    new_tensor = mask + (1 - mask) * tensor
    new_tensor = torch_to_np(new_tensor)
    imageio.imsave(out_name, new_tensor)
    return new_tensor


def imgs_dir_to_mp4(in_query, output_filename, total_files):
    os.system('ffmpeg -y -framerate 10 -i %s -b:v 2M -vcodec msmpeg4 -acodec wmav2 -loglevel error '
              '-r 30 -filter_complex loop=loop=2:size=%d:start=0 %s'
              % (in_query, total_files, output_filename))


def log_photos(batch_h, outdir):
    photo_maps = batch_h.copy()

    # NOTE: this assume photo_maps to be a dict with pairs key:map
    # where map is a single image. Therefore, batch_size for the photos
    # is expected to be 1
    photo_maps_keys = photo_maps.keys()
    for k in photo_maps_keys:
        if k in ['name', 'mask', 'light', 'light_pos', 'light_neq', 'tport', 'tport_specular'] or 'res_' in k:
            continue

        photo_maps[k] = torch.cat([photo_maps[k], photo_maps['mask']], dim=1)

    if 'img' in photo_maps_keys:
        image = photo_maps['img'].squeeze().clamp(0, 1)
        figname = '%s/input_img.png' % (outdir)
        imageio.imsave(figname, torch_to_np(image).astype(np.uint8), compression=3)
    if 'mask' in photo_maps_keys:
        mask = photo_maps['mask'].squeeze()
        figname = '%s/mask.png' % (outdir)
        imageio.imsave(figname, torch_to_np(mask.unsqueeze(0)).astype(np.uint8), compression=3)
    if 'img_sh' in photo_maps_keys:
        image = photo_maps['img_sh'].squeeze().clamp(0, 1)
        figname = '%s/img_sh.png' % (outdir)
        imageio.imsave(figname, torch_to_np(image), compression=3)
    if 'albedo' in photo_maps_keys:
        albedo = photo_maps['albedo'].squeeze().clamp(0, 1)
        figname = '%s/pred_albedo.png' % (outdir)
        imageio.imsave(figname, torch_to_np(albedo), compression=3)
    if 'shading' in photo_maps_keys:
        image = photo_maps['shading'].squeeze()
        image[:3] = (image[:3] + 1e-8) / image[:3].max()
        figname = '%s/pred_shading_relight.png' % (outdir)
        imageio.imsave(figname, torch_to_np(image), compression=3)
    if 'residual' in photo_maps_keys:
        photo_maps['residual'] = (photo_maps['residual'] * photo_maps['mask'])
        photo_maps['residual'] /= photo_maps['residual'].max()
        image = photo_maps['residual'].squeeze() * 10
        image = (image ** (1 / 2.2)).clamp(0, 1)
        figname = '%s/residual.png' % (outdir)
        imageio.imsave(figname, torch_to_np(image), compression=3)
    if 'tport' in photo_maps_keys:
        tport = photo_maps['tport'].clamp(-1, 1).detach().cpu().squeeze().float()
        figname = '%s/pred_tport.pth' % (outdir)
        torch.save(tport, figname)

        tport_pos = torch_to_np(tport.clamp(min=0))
        tport_neq = torch_to_np(tport.clamp(max=0).abs())

        tport_pos_dir = os.path.join(outdir, 'tport_pos')
        tport_neq_dir = os.path.join(outdir, 'tport_neq')

        os.makedirs(tport_pos_dir, exist_ok=True)
        os.makedirs(tport_neq_dir, exist_ok=True)
        for i in range(tport_pos.shape[-1]):
            figname = '%s/pred_tport_pos_%d.png' % (tport_pos_dir, i)
            imageio.imsave(figname, tport_pos[..., i], compression=3)
            figname = '%s/pred_tport_neq_%d.png' % (tport_neq_dir, i)
            imageio.imsave(figname, tport_neq[..., i], compression=3)
    if 'tport_residual' in photo_maps_keys:
        tport = photo_maps['tport_residual'].clamp(-1, 1).detach().cpu().squeeze().float()
        figname = '%s/pred_tport_residual.pth' % (outdir)
        torch.save(tport, figname)

        tport_pos = torch_to_np(tport * (tport > 0))
        tport_neq = torch_to_np(torch.abs(tport * (tport < 0)))

        tport_pos_dir = os.path.join(outdir, 'tport_specular_pos')
        tport_neq_dir = os.path.join(outdir, 'tport_specular_neq')

        os.makedirs(tport_pos_dir, exist_ok=True)
        os.makedirs(tport_neq_dir, exist_ok=True)
        for i in range(tport_pos.shape[-1]):
            figname = '%s/pred_tport_specular_pos_%d.png' % (tport_pos_dir, i)
            imageio.imsave(figname, tport_pos[..., i].astype(np.uint8), compression=3)
            figname = '%s/pred_tport_specular_neq_%d.png' % (tport_neq_dir, i)
            imageio.imsave(figname, tport_neq[..., i].astype(np.uint8), compression=3)
    if 'light' in photo_maps_keys:
        light = photo_maps['light'].squeeze().detach().cpu().float()
        figname = '%s/pred_light.exr' % (outdir)
        size = int(math.sqrt(light.numel() / 3))
        imageio.imwrite(figname, light.view(size, size, 3).numpy())
        figname = '%s/pred_light.pth' % (outdir)
        torch.save(light, figname)
    if 'img_sh_noresidual' in photo_maps_keys:
        image = photo_maps['img_sh_nospecular'].squeeze().clamp(0, 1)
        figname = '%s/img_sh_nospecular.png' % (outdir)
        imageio.imsave(figname, torch_to_np(image), compression=3)


def relight_photos(model, photos_dir, light_dir, output_folder):
    # create photos dataset
    photos_data = PhotosDataset(photos_dir)

    # load target light
    ncoeffs = model.hparams.ncoeffs
    light_name = os.path.basename(light_dir)
    light_coeffs = torch.stack(load_light_from_dir(light_dir, ncoeffs)).to(model.device, model.dtype)

    for batch in photos_data:
        # move the batch to the correct device
        for k, v in batch.items():
            if k != 'name':
                batch[k] = batch[k].to(model.device, model.dtype).unsqueeze(0)

        # forward it through the model and estimate the
        # intermediate layers: albedo, transport, residual...
        batch_h = model.forward_photo(batch)

        # create folder structure
        human_folder = os.path.join(output_folder, batch['name'])
        relighted_root = os.path.join(human_folder, 'relighted', light_name)
        relighted_folder = relighted_root + '/image'
        shading_folder = relighted_root + '/shading'
        residual_folder = relighted_root + '/residual'
        os.makedirs(human_folder, exist_ok=True)
        os.makedirs(relighted_folder, exist_ok=True)
        os.makedirs(shading_folder, exist_ok=True)
        os.makedirs(residual_folder, exist_ok=True)

        # store intermediate results
        log_photos(batch_h, human_folder)

        # relight the input image with the given light
        total_files = 0
        init_ix = 10
        final_ix = 21
        total = 60 - (init_ix + final_ix) - 1

        for i, light_rot in tqdm(enumerate(light_coeffs), total=len(light_coeffs)):
            # avoid relighting when the light is behind
            if i <= init_ix or (len(light_coeffs) - final_ix) <= i:
                continue

            # compute shading and residual, and relighted result
            shading = model.render_shading(batch_h['tport'], light_rot.unsqueeze(0))
            residual = model.render_shading(batch_h['tport_residual'], light_rot.unsqueeze(0)) * batch['mask']
            img_relighted = model.render_img(batch_h['albedo'], shading, residual) * batch['mask']

            # modify residual for visualization purposes
            residual = residual / residual.max()
            residual = (residual * 10 ** (1 / 2.2)).clamp(0, 1)

            # make a loop: compute indexes to store images
            # note that the first relighted image is also the last
            total_files += 2
            offset_init = (i - init_ix)
            offset_end = total * 2 - offset_init - 1

            shading = torch.cat([shading, batch['mask']], dim=1)[0]
            store_tensor_to_img(shading, shading_folder + '/%02d.png' % offset_init)
            store_tensor_to_img(shading, shading_folder + '/%02d.png' % offset_end)

            residual = torch.cat([residual, batch['mask']], dim=1)[0]
            store_tensor_to_img(residual, residual_folder + '/%02d.png' % offset_init)
            store_tensor_to_img(residual, residual_folder + '/%02d.png' % offset_end)

            img_relighted = torch.cat([img_relighted, batch['mask']], dim=1)[0]
            store_tensor_to_img(img_relighted, relighted_folder + '/%02d.png' % offset_init)
            store_tensor_to_img(img_relighted, relighted_folder + '/%02d.png' % offset_end)

        # store gifs from the images in each of the dirs
        path = relighted_root + '/%s-shading.wmv' % batch['name']
        imgs_dir_to_mp4(shading_folder + '/%02d.png', path, total_files)

        path = relighted_root + '/%s-residual.wmv' % batch['name']
        imgs_dir_to_mp4(residual_folder + '/%02d.png', path, total_files)

        path = relighted_root + '/%s-relighted.wmv' % batch['name']
        imgs_dir_to_mp4(relighted_folder + '/%02d.png', path, total_files)


if __name__ == '__main__':
    ckpt_dir = './data/model/relighting_model.ckpt'
    ckpt_args = './data/model/hparams.yaml'

    photos_dir = './data/photos'
    light_dir = './data/lights/pisa'

    out_relighting_dir = './data/photos_relighted'
    os.makedirs(out_relighting_dir, exist_ok=True)

    # load and init the model
    hparams = AttributeDict(load_hparams_from_yaml(ckpt_args))
    model = SigasiaSystem.load_from_checkpoint(
        checkpoint_path=ckpt_dir,
        hparams=hparams
    )
    model.eval()

    # run the relighting algorithm
    with torch.set_grad_enabled(False):
        relight_photos(model, photos_dir, light_dir, out_relighting_dir)
