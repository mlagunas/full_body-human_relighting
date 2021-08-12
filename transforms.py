import torch


def get_bbox(mask, pad=0):
    """
    return corners of the smallest bounding box that
        includes all the masks in the batch
    """
    mask = mask.squeeze().to(dtype=torch.uint8)
    if len(mask.shape) == 2:  # if it is a non-batched mask
        mask = mask.unsqueeze(0)
    assert len(mask.shape) == 3, 'mask shape is: %s' % str(mask.shape)

    rows = torch.any(mask, axis=1)
    cols = torch.any(mask, axis=2)
    in_mask_col = torch.nonzero(rows, as_tuple=True)[1]
    in_mask_row = torch.nonzero(cols, as_tuple=True)[1]

    rmin = in_mask_row.min()
    rmax = in_mask_row.max()
    cmin = in_mask_col.min()
    cmax = in_mask_col.max()

    if isinstance(pad, float):
        pad_r = int(pad * (rmax - rmin))
        pad_c = int(pad * (cmax - cmin))
    elif isinstance(pad, int):
        pad_r = pad_c = pad

    rmin = (rmin - pad_r).item()
    rmax = (rmax + pad_r).item()
    cmin = (cmin - pad_c).item()
    cmax = (cmax + pad_c).item()

    rmin = max(rmin, 0)
    rmax = min(rmax, mask.shape[1])
    cmin = max(cmin, 0)
    cmax = min(cmax, mask.shape[2])

    if (rmax - rmin) % 2 != 0:
        if rmax < mask.shape[1]:
            rmax += 1
        elif rmin > 0:
            rmin -= 1
    if (cmax - cmin) % 2 != 0:
        if cmax < mask.shape[1]:
            cmax += 1
        elif cmin > 0:
            cmin -= 1

    return rmin, rmax, cmin, cmax


def roi_crop(batch, dict_keys=None, pad=20):
    """
    crops the region of interest in the batch.
        The batch should be a dict with a "mask" key
    """
    if dict_keys is None:
        dict_keys = list(batch.keys())

    if 'input_mask' in dict_keys:
        mask_key = 'input_mask'
    elif 'mask' in dict_keys:
        mask_key = 'mask'
    else:
        return batch

    _device = batch[mask_key].device
    _dtype = batch[mask_key].dtype

    rmin, rmax, cmin, cmax = get_bbox(batch[mask_key], pad=0)
    for key in dict_keys:
        if 'coeffs' in key or 'light' in key or 'exp' == key or 'name' == key:
            continue

        shape = batch[key].shape
        h = rmax - rmin
        w = cmax - cmin

        b_key = torch.zeros(*shape[:-2], h + (2 * pad), w + (2 * pad), device=_device, dtype=_dtype)
        b_key[..., pad:-pad, pad:-pad] = batch[key][..., rmin:rmax, cmin:cmax]
        batch[key] = b_key

    return batch


def to_tensor(np_img):
    if isinstance(np_img, torch.Tensor):
        return np_img

    # if it is an image reshape dims to match pytorch standards
    if len(np_img.shape) == 3:
        return torch.as_tensor(np_img.copy()).permute(2, 0, 1).contiguous()
    if len(np_img.shape) == 4:
        return torch.as_tensor(np_img.copy()).permute(0, 3, 1, 2).contiguous()

    # if it is not an image just transform to pytorch
    return torch.as_tensor(np_img.copy()).contiguous()
