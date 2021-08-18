# Single-image Full-body Human Relighting
Implementation of the paper: Single-image Full-body Human Relighting

### Requirements  

- First make sure that you have Pytorch running in your machine: https://pytorch.org/ (tested with version 1.9).
- Install all the python dependencies with `pip install -r requirements.txt`.
- You will need `ffmpeg` in order to generate the relighted videos: `sudo apt install ffmpeg`.
- Download the pretrained model from XX and place it under `./data/model/`.

### Relighting your photos

Before running `photo_relighting.py`:
- You can change the lights and the photos to use by modifying the following lines:
```
photos_dir = './data/photos'
light_dir = './data/lights/pisa'
```

### Things to be done

- Upload the training code.
- Add script to generate your own light coefficients from any input image in lat-long format.


### Citation

If you find this code useful please cite our work with:
```
@inproceedings{Lagunas2021humanrelighting,
    title={Single-image Full-body Human Relighting},
    booktitle={Eurographics Symposium on Rendering (EGSR)},
    publisher={The Eurographics Association},
    author={Lagunas, Manuel and Sun, Xin and Yang, Jimei and Villegas, Ruben and Zhang, Jianming and Shu, Zhixin and Masia, Belen and Gutierrez, Diego},
    year={2021},
    DOI = {10.2312/sr.20211301}
}
```

