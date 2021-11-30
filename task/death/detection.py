import warnings as wrng
from typing import Optional, Sequence, Tuple, Union

import cell_tracking_BC.task.pattern as ptmt
import numpy as nmpy
import task.death.filters as fltr
from scipy.signal import medfilt as MedianFiltered


array_t = nmpy.ndarray


def FilterForEntropy() -> Tuple[array_t, float]:
    """"""
    return fltr.Sigmoid_Inv(-100, 100, 25, 1.5), 1.0


def FilterForVariance() -> Tuple[array_t, float]:
    """"""
    return fltr.Sigmoid(-100, 100, 25, 1.5), 1.0e10


def PredictedDeathTime(
    feature: Union[array_t, Sequence[float]],
    first_frm: int,
    last_frm: int,
    lower_bound: float,
    last_div_frm: int,
    filter_w_scaling: Tuple[array_t, float],
) -> Tuple[Optional[int], Union[array_t, Sequence[float]]]:
    """"""
    smth_feature = MedianFiltered(MedianFiltered(feature, 15), 15)
    filter_, feature_scaling = filter_w_scaling

    # Truncate the trajectory after the last division event, if any
    if last_div_frm > 0:
        smth_feature = smth_feature[last_div_frm:]
        if smth_feature.size < filter_.size:
            wrng.warn("Short signal after truncation; Expect template matching failure")
    smth_feature = smth_feature / feature_scaling

    matching_response = ptmt.match_template(
        smth_feature, filter_, normalized=None, pad_input=True, mode="edge"
    )
    if matching_response is None:
        return None, (-1.0,)

    max_response = max(matching_response)
    where_max = nmpy.where(matching_response == max_response)[0][0]

    if max_response > lower_bound:
        dth_frm = last_div_frm + where_max

        # Discard incomplete events occurring at start or end of sequence
        if first_frm < dth_frm < last_frm:
            return dth_frm, matching_response
        else:
            return None, matching_response
    else:
        return None, matching_response
