import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from transforms import roi_crop
from relighting_unet import UNet


class SigasiaSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(SigasiaSystem, self).__init__()

        # read experiment params
        self.save_hyperparameters(hparams)

        self.model = UNet(hparams)
        if hparams.init_weights:
            self.model.init_weights()

        # to plot the parameters
        self.example_input_array = [{
            'img_sh': torch.rand(2, 3, 256, 256).clamp(-1, 1),
            # 'img_pt': torch.rand(2, 3, 256, 256).clamp(-1, 1),
            'mask': torch.rand(2, 1, 256, 256).clamp(-1, 1),
            'light': torch.rand(2, 9, 3).clamp(-1, 1),
            'target_light': torch.rand(2, 9, 3).clamp(-1, 1)
        }]

    def forward(self, batch):
        tport, (tport_diffuse_pos, tport_diffuse_neq), \
        tport_specular, \
        light, (light_pos, light_neq), \
        albedo = \
            self.model.forward(2 * batch['img_sh'] - 1)

        # reconstruct images
        shading = self.render_shading(tport, light)
        specular = self.render_shading(tport_specular, light)
        img_sh = self.render_img(albedo, shading, specular)

        # build dict object and return all the maps
        to_return = {
            'img_sh': img_sh * batch['mask'],
            'albedo': albedo * batch['mask'],
            'shading': shading * batch['mask'],
            'residual': specular * batch['mask'],

            'tport': tport * batch['mask'],
            'tport_pos': tport_diffuse_pos * batch['mask'],
            'tport_neq': tport_diffuse_neq * batch['mask'],

            'tport_residual': tport_specular * batch['mask'],

            'light': light,
            'light_pos': light_pos,
            'light_neq': light_neq,
        }

        return to_return

    def forward_photo(self, batch):
        for k in batch.keys():
            if 'name' not in k:
                if len(batch[k].shape) == 3:
                    batch[k] = batch[k].unsqueeze(0).to(self.device)

        batch['img_sh'] *= batch['mask']
        batch_hat = self.forward(batch)
        batch_hat['img_sh_nospecular'] = self.render_img(batch_hat['shading'], batch_hat['albedo'])
        return {
            'name': batch['name'],
            'img': batch['img_sh'],
            'mask': batch['mask'],
            **batch_hat  # add the predicted maps to be stored also
        }

    def render_full(self, tport, albedo, light, specular, exp=None):
        shading = self.render_shading(tport, light, exp=exp)
        img_sh = self.render_img(albedo, shading, specular)
        return shading, img_sh

    def render_shading(self, tport, light, exp=None):

        shading = (tport.unsqueeze(1) * light.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        return shading.clamp(min=0, max=10)

    @staticmethod
    def render_img(albedo: torch.Tensor, shading: torch.Tensor, specular=None) -> torch.Tensor:
        if specular is None:
            return (albedo * shading).clamp(0, 1)
        else:
            return ((albedo * shading) + specular).clamp(0, 1)

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        elif self.hparams.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, betas=(0.5, 0.999))
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9,
                                        weight_decay=5e-4, nesterov=True)
        else:
            raise ValueError('--optimizer should be one of [sgd, adam]')

        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=5,
                factor=0.1),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def eval(self):
        super().eval()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # we perform batch norm without using
                # a running mean and std
                m.train()
        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-weights', default=True, type=bool)
    parser.add_argument('--ncoeffs', default=9, type=int)
    parser.add_argument('--encoder-channels', default=[3, 64, 128, 256, 512], type=int)
    parser.add_argument('--decoder-channels', default=[512, 256, 128, 64, 32], type=int)
    parser.add_argument('--light-channels', default=[512 * 3, 512, 256, 128], type=int)
    parser.add_argument('--n-res-layers', default=2, type=int)
    parser.add_argument('--act', default='relu', type=str)
    parser.add_argument('--norm', default='none', type=str)
    parser.add_argument('--upsample', default='bilinear', type=str)
    parser.add_argument('--layers-with-dropout', default=2, type=str)
    hparams = parser.parse_args()

    img = torch.rand(4, 3, 256, 512)
    target_light = torch.rand(4, 9 * 3)
    model = UNet(hparams)
    model.forward(img, target_light)
