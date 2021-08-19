import torch
import torch.nn as nn
from torch.nn import functional as F

""" Auxiliary functions and classes
"""


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight, 0, 0.02)


def activation_func(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    if activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    if activation == 'selu':
        return nn.SELU(inplace=True)
    if activation == 'none':
        return nn.Identity()


def normalization_func(normalization):
    if normalization == 'batch':
        return lambda ch: nn.BatchNorm2d(ch, momentum=0.5)
    if normalization == 'batch_no_stats':
        return lambda ch: nn.BatchNorm2d(ch, track_running_stats=False)
    if normalization == 'instance':
        return lambda ch: nn.InstanceNorm2d(ch)
    if normalization == 'none':
        return lambda _: nn.Identity()


def conv4x4(in_ch, out_ch, stride=2, padding=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=padding, bias=bias)


def conv3x3(in_ch, out_ch, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding, bias=bias)


class AdaptiveConcat:
    def __call__(self, x1, x2):
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        ch = hparams.encoder_channels[-1]
        norm_fn = normalization_func(hparams.norm)
        act_fn = activation_func(hparams.act)
        use_bias = norm_fn is None

        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=use_bias)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=use_bias)
        self.relu = act_fn

        self.b1 = norm_fn(ch)
        self.b2 = norm_fn(ch)

    def forward(self, x):
        residual = x
        x = self.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        x = residual + x
        return self.relu(x)


class DoubleConv(nn.Module):

    def __init__(self, hparams, in_channels, out_channels, mid_channels=None):
        super().__init__()

        norm_fn = normalization_func(hparams.norm)
        act_fn = activation_func(hparams.act)
        use_bias = norm_fn is None

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_fn(mid_channels),
            act_fn,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            act_fn
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    def __init__(self, hparams, in_channels, out_channels, add_dropout=False):
        super().__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.add_dropout = add_dropout
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(hparams, in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.up(x)

        x = self.double_conv(x)

        if self.add_dropout:
            x = self.dropout(x)

        return x


class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        self.encoder = self._make_layers(hparams)

    def _make_layers(self, hparams):
        # get custom fns
        norm_fn = normalization_func(hparams.norm)
        act_fn = activation_func(hparams.act)
        use_bias = hparams.norm is None

        encoder_layers = []
        for i in range(len(hparams.encoder_channels) - 1):
            ch0 = hparams.encoder_channels[i]
            ch1 = hparams.encoder_channels[i + 1]

            conv = conv3x3(ch0, ch1, stride=2, padding=1, bias=use_bias)
            bn = norm_fn(ch1)
            relu = act_fn
            encoder_layers.append(nn.Sequential(conv, bn, relu))

        return nn.ModuleList(encoder_layers)

    def forward(self, x):
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)

        return encoder_features


class Decoder(nn.Module):
    def __init__(self, hparams, out_ch=3):
        super(Decoder, self).__init__()

        self.cat = AdaptiveConcat()
        self.dropout = nn.Dropout(p=0.5)
        self.decoder = self._make_layers(hparams)
        self.lastconv = nn.Conv2d(hparams.decoder_channels[-1], out_ch, kernel_size=1, bias=True)

    def _make_layers(self, hparams):
        decoder_layers = []
        for i in range(len(hparams.decoder_channels) - 1):
            ch0 = hparams.decoder_channels[i]
            ch1 = hparams.decoder_channels[i + 1]

            # add dropout layers if needed
            should_add_dropout = i <= hparams.layers_with_dropout
            decoder_layers.append(UpBlock(hparams, ch0 * 2, ch1, add_dropout=should_add_dropout))

        return nn.ModuleList(decoder_layers)

    def forward(self, residual_features, encoder_features):
        x = residual_features
        for i, layer in enumerate(self.decoder):
            x = layer(x, encoder_features[-(i + 1)])

        return self.lastconv(x)


class Residual(nn.Module):
    def __init__(self, hparams):
        super(Residual, self).__init__()
        self.residual = self._make_residual_layers(hparams)

    def forward(self, x):
        return self.residual(x)

    def _make_residual_layers(self, hparams):
        residual_layers = []
        for i in range(hparams.n_res_layers):
            residual_layers.append(ResidualBlock(hparams))

        return nn.Sequential(*residual_layers)


class MLPlight(nn.Module):
    def __init__(self, hparams):
        super(MLPlight, self).__init__()
        self.mlp = self._make_layers(hparams)

    def forward(self, *all_features):
        features = torch.cat(*all_features, dim=1)
        return self.mlp(features)

    def _make_layers(self, hparams):
        norm_fn = normalization_func(hparams.norm)
        act_fn = activation_func(hparams.act)
        use_bias = hparams.norm is None

        # add convolutional layers to the model that reduce dimension by 2
        mlp_layers = []
        for i in range(len(hparams.light_channels) - 1):
            ch0 = hparams.light_channels[i]
            ch1 = hparams.light_channels[i + 1]

            conv = conv3x3(ch0, ch1, stride=2, padding=1, bias=use_bias)
            bn = norm_fn(ch1)
            relu = act_fn
            mlp_layers.extend([conv, bn, relu])
            if i <= hparams.layers_with_dropout:
                mlp_layers.append(nn.Dropout(0.5))

        # make one dimensional and output correct number of features
        mlp_layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        mlp_layers.append(nn.Flatten())
        mlp_layers.append(nn.Linear(hparams.light_channels[-1], hparams.ncoeffs * 3, bias=True))

        return nn.Sequential(*mlp_layers)


class DecoderLight(nn.Module):
    def __init__(self, hparams):
        super(DecoderLight, self).__init__()
        self.mlp = self._make_layers(hparams)

    def forward(self, target_light):
        if len(target_light.shape) > 2:
            target_light = target_light.view(target_light.shape[0], -1)
        return self.mlp(target_light)

    def _make_layers(self, hparams):
        act_fn = activation_func(hparams.act)

        mlp_layers = []
        mlp_layers.append(nn.Linear(hparams.ncoeffs * 3, hparams.encoder_channels[-3]))
        mlp_layers.append(act_fn)
        mlp_layers.append(nn.Linear(hparams.encoder_channels[-3], hparams.encoder_channels[-2]))
        mlp_layers.append(act_fn)
        mlp_layers.append(nn.Linear(hparams.encoder_channels[-2], hparams.encoder_channels[-1]))
        return nn.Sequential(*mlp_layers)


""" Full model class
"""


class UNet(nn.Module):

    def __init__(self, hparams):
        super(UNet, self).__init__()

        self.encoder = Encoder(hparams)

        self.residual_tport_diffuse = Residual(hparams)
        self.residual_tport_specular = Residual(hparams)

        self.decoder_tport_diffuse_pos = Decoder(hparams, out_ch=hparams.ncoeffs)
        self.decoder_tport_diffuse_neq = Decoder(hparams, out_ch=hparams.ncoeffs)
        self.reconst_tport_diffuse = nn.Conv2d(hparams.ncoeffs, hparams.ncoeffs, kernel_size=1, bias=True)

        self.decoder_tport_specular_pos = Decoder(hparams, out_ch=hparams.ncoeffs)
        self.decoder_tport_specular_neq = Decoder(hparams, out_ch=hparams.ncoeffs)
        self.reconst_tport_specular = nn.Conv2d(hparams.ncoeffs, hparams.ncoeffs, kernel_size=1, bias=True)

        self.decoder_light_pos = MLPlight(hparams)
        self.decoder_light_neq = MLPlight(hparams)
        self.reconst_light = nn.Linear(hparams.ncoeffs * 3, hparams.ncoeffs * 3, bias=True)

        hparams.norm = 'batch'
        self.residual_albedo = Residual(hparams)
        self.decoder_albedo = Decoder(hparams, out_ch=3)

    def init_weights(self):
        self.encoder.apply(weights_init_normal)

        self.decoder_tport_diffuse_pos.apply(weights_init_normal)
        self.decoder_tport_diffuse_neq.apply(weights_init_normal)

        self.decoder_tport_specular_pos.apply(weights_init_normal)
        self.decoder_tport_specular_neq.apply(weights_init_normal)

        self.decoder_albedo.apply(weights_init_normal)

        self.decoder_light_pos.apply(weights_init_normal)
        self.decoder_light_neq.apply(weights_init_normal)

        self.residual_tport_diffuse.apply(weights_init_normal)
        self.residual_tport_specular.apply(weights_init_normal)
        self.residual_albedo.apply(weights_init_normal)

    def forward(self, x):
        x_enc = self.get_skip_connections(x)
        return self.forward_decoder(x, x_enc)

    def get_skip_connections(self, x):
        self.skip_connections = self.encoder(x)
        return self.skip_connections

    def forward_decoder(self, x, x_enc):
        # first forward through the residual parts of each sub-block
        res_tport_diffuse = self.residual_tport_diffuse(x_enc[-1])
        res_tport_specular = self.residual_tport_specular(x_enc[-1])
        res_albedo = self.residual_albedo(x_enc[-1])

        # forward the residual parts through the reconstruction sections
        albedo = (self.decoder_albedo(res_albedo, x_enc) + x).clamp(0, 1)

        tport_diffuse_pos = self.decoder_tport_diffuse_pos(res_tport_diffuse, x_enc)
        tport_diffuse_neq = self.decoder_tport_diffuse_neq(res_tport_diffuse, x_enc)
        tport_diffuse = self.reconst_tport_diffuse(tport_diffuse_pos - tport_diffuse_neq)

        tport_specular_pos = self.decoder_tport_specular_pos(res_tport_specular, x_enc)
        tport_specular_neq = self.decoder_tport_specular_neq(res_tport_specular, x_enc)
        tport_specular = self.reconst_tport_diffuse(tport_specular_pos - tport_specular_neq)

        # now that we have reconstructed the albedo
        # and transport use the intermediate features to
        # reconstruct the light
        all_features = [res_tport_diffuse, res_tport_specular, res_albedo, x_enc[-1]]
        light_pos = self.decoder_light_pos(all_features)
        light_neq = self.decoder_light_neq(all_features)
        light = self.reconst_light(light_pos - light_neq)

        light_pos = light_pos.view(light_pos.shape[0], 3, tport_diffuse_pos.shape[1])
        light_neq = light_neq.view(light_neq.shape[0], 3, tport_diffuse_pos.shape[1])
        light = light.view(light.shape[0], 3, tport_diffuse_neq.shape[1], )

        return tport_diffuse, (tport_diffuse_pos, tport_diffuse_neq), \
               tport_specular, \
               light, (light_pos, light_neq), \
               albedo,


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--init-weights', default=True, type=bool)
    parser.add_argument('--ncoeffs', default=9, type=int)
    parser.add_argument('--encoder-channels', default=[3, 64, 128, 256, 512], type=int)
    parser.add_argument('--decoder-channels', default=[512, 256, 128, 64, 32], type=int)
    parser.add_argument('--light-channels', default=[512 * 4, 512, 256, 128], type=int)
    parser.add_argument('--n-res-layers', default=2, type=int)
    parser.add_argument('--act', default='relu', type=str)
    parser.add_argument('--norm', default='none', type=str)
    parser.add_argument('--upsample', default='bilinear', type=str)
    parser.add_argument('--layers-with-dropout', default=2, type=str)
    hparams = parser.parse_args()

    img = torch.rand(4, 3, 256, 512)
    model = UNet(hparams)
    out = model.forward(img)
    print(len(out))
