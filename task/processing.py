from typing import Dict, Sequence

import numpy as nmpy
import scipy.signal as sgnl
import skimage.exposure as xpsr
import skimage.filters as fltr


def RescaledPOLChannel(channels: Dict[str, nmpy.ndarray]) -> nmpy.ndarray:
    """"""
    raw_frame = channels["POL"]
    in_range = nmpy.around(nmpy.percentile(raw_frame, (5, 95))).astype(nmpy.uint64)
    raw_frame = xpsr.rescale_intensity(raw_frame, in_range=tuple(in_range))
    output = 255.0 * raw_frame

    return output


def ContrastNormalized(frame: nmpy.ndarray, percentile: Sequence[int], /) -> nmpy.ndarray:
    """
    TODO: Several problems:
        - edges: not edges since the filter is a smoothing filter
        - why smoothing the frame after rescaling intensity?
    """
    kernel = nmpy.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kernel = kernel / nmpy.sum(kernel)

    edges = sgnl.convolve2d(frame, kernel, mode="same")

    p_inf = nmpy.percentile(edges, percentile[0])
    p_sup = nmpy.percentile(edges, percentile[1])
    img = xpsr.rescale_intensity(frame, in_range=(p_inf, p_sup))

    smooth_frm = fltr.gaussian(img, sigma=3, multichannel=None)

    return smooth_frm
