import imageio
import os

if __name__ == '__main__':
    input_image_path = 'data/photos_removebg/diego.png'
    out_dir = 'data/photos'

    img_rgba = imageio.imread(input_image_path)
    assert img_rgba.shape[-1] == 4, 'img is not RGBA'
    img_name = os.path.basename(input_image_path).split('.')[0]

    img_a = img_rgba[..., -1:]
    img_rgb = img_rgba[..., :-1]

    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'original'), exist_ok=True)

    imageio.imsave(os.path.join(out_dir, 'mask', img_name + '.png'), img_a)
    imageio.imsave(os.path.join(out_dir, 'original', img_name + '.png'), img_rgb)
