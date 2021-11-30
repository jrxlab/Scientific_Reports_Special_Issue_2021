# Dynamic analysis framework to detect cell division and cell death on label-free microscopy time-lapses

The proposed dynamic analysis framework is a computational pipeline which runs an automatic analysis on a live-cell, label-free microscopy time-lapse to detect cell divisions and cell deaths. It requires two types of channel: fluorescent channels to segment (i.e. find the regions of) the objects of interest (nuclei, cytoplasms, or cells), and a polarized channel to detect cell death events.

___
*Note that the pipeline is fully functional for a single type of object of interest, typically cells. The joint handling of several types of object, typically nuclei and cells, is only partial at the moment.*
___



## Overview

The following description focuses on cells (see *Note* above about also dealing with nuclei or cytoplasms). The main steps of the analysis are:

- Segment the cells on the successive frames of the appropriate fluorescent channel using a convolutional neural network (CNN).
- Starting from the segmentation of the first frame, track the position of each cell from a segmented frame to the next, taking into account cell divisions, in order to build its tree-shaped, XYT-trajectory. Note that this trajectory tree might be just a trunk.
- In each trajectory, detect **cell divisions** as branch creations.
- Along each trajectory, extract the signal within the cell on the corresponding frames of the polarized channel, and *summarize* it into a scalar value to build a time series.
- Apply pattern matching to each time series, and detect **cell death** as the time-point where the matching score is above a given threshold.



## Details

The CNN currently used for cell segmentation comes from the U-Net publication (see Reference below). Switching to another architecture, say cellpose (see Reference below), to adapt the pipeline to various cell types is only a matter of changing a parameter of the pipeline.

The cell death event detection currently use the following elements:

- The polarized signal "summarizing" value is the Shannon entropy. (The signal variance has also been tested as an alternative, but revealed to be a bit less accurate for our purpose.)

- The cell death pattern is an inverted (i.e. decreasing) sigmoid function (see Reference below).



## Implementation

The programming language of the pipeline is Python 3. It relies on several open-source libraries, notably ``cell_tracking_BC``. The overall implementation structure is:

- The main entry point is the ``run_pipeline.py`` script. It loads the pipeline parameters from ``run_parameters.py``.

- The U-Net network used for cell segmentation can be found in the folder ``hela_p53_c8_cell``. Its path is specified in ``run_parameters.py``.

- The ``task`` folder contains the Python modules for frame processing (used in pre-processing steps for segmentation) and cell death detection (pattern definition and pattern matching).



## References

Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI, 2015.

[Cellpose](https://cellpose.readthedocs.io/en/latest/)

[Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)

[cell_tracking_BC](https://pypi.org/project/cell-tracking-bc/)