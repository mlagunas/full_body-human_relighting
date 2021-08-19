# Single-image Full-body Human Relighting
Implementation of the paper: Single-image Full-body Human Relighting

### Requirements  

- First make sure that you have Pytorch running in your machine: https://pytorch.org/ (tested with version 1.9)
- Install all the python dependencies with `pip install -r requirements.txt`
- You will need `ffmpeg` in order to generate the relighted videos: `sudo apt install ffmpeg`
- Download the pretrained model from [here](https://drive.google.com/file/d/13BZ_etfYeXTCCMr2-Hg8EVKDCDv7Y_YC/view?usp=sharing) and place it under `./data/model/`

_Note that this code has been tested out using Ubuntu 20.04 and Python 3.8_

### Relighting your photos

Before running `photo_relighting.py`:
- You can change the lights and the photos to use by modifying the following lines:
```
photos_dir = './data/photos'
light_dir = './data/lights/pisa'
```

Note that the `photos` folder has the following structure:
```
/data/
 |  photos/
 |  |  mask/
 |  |  |  your_photo.png 
 |  |  original/
 |  |  |  your_photo.png
```

If you want to relight your own images, make sure that they follow the aforementioned structure. To extract the mask from your photographs, you can rely on freely available services such as [that one](https://www.remove.bg/). 

Note that both the mask and the original image should have the same spatial resolution. You can use the script `removebg_img_split.py` to automatically split the image you downloaded with the masked background. Make sure that you correctly set the `img_path` and `out_dir` variables in the script.



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

