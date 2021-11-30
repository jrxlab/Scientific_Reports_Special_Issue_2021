import sys as sstm
import warnings as wrng

import better_exceptions as xcpt
import cell_tracking_BC.in_out.file.cache as cche
import cell_tracking_BC.in_out.file.feature as ftst
import cell_tracking_BC.in_out.file.sequence as sqio
import cell_tracking_BC.in_out.graphics.segmentation as gvwr
import cell_tracking_BC.in_out.graphics.sequence as qvwr
import cell_tracking_BC.in_out.graphics.tracking as tvwr
import cell_tracking_BC.task.feature as ftre
import cell_tracking_BC.task.processing as prss
import cell_tracking_BC.task.segmentation as sgmt
import cell_tracking_BC.task.tracking.main as trck
import numpy as nmpy
import run_parameters as prmt
import task.death.detection as death
from cell_tracking_BC.in_out.graphics.dbe.matplotlib.any_d import figure_t as m_figure_t
from cell_tracking_BC.in_out.graphics.dbe.vedo.any_d import figure_t as v_figure_t
from cell_tracking_BC.in_out.graphics.dbe.vedo.context import (
    DRAWING_CONTEXT as DRAWING_CONTEXT_VEDO,
)
from cell_tracking_BC.in_out.text.logger import LOGGER
from cell_tracking_BC.task.tracking.tracker import gmb_tracker_t as tracker_t
from cell_tracking_BC.type.segmentations import segmentations_t
from cell_tracking_BC.type.sequence import sequence_t
from task.processing import ContrastNormalized, RescaledPOLChannel


xcpt.hook()

if prmt.caching:
    cache = cche.CacheDictionaryForMain(
        __file__,
        folder_postfix="-SKIP",
        cache_postfix=f"-Seq{prmt.sequence_number}-From{prmt.first_frame}",
    )
else:
    cache = {}  # Dummy cache


LOGGER.info(f"--- READING SEQUENCE: {prmt.sequence_path}")
frames = sqio.SequenceFromPath(prmt.sequence_path)
sequence = sequence_t.NewFromFrames(
    frames,
    prmt.sequence_channels,
    prmt.sequence_path,
    first_frame=prmt.first_frame,
    last_frame=prmt.last_frame,
)

print("+++ Original:", sequence, sep="\n")
if prmt.show_sequence:
    qvwr.ShowSequence(sequence)


LOGGER.info(f"--- REGISTRATION on {prmt.registration_channel}: {prmt.register_channels}")
if prmt.register_channels:
    sequence.ApplyTransform(
        prss.RegisteredInTranslation,
        reference=prmt.registration_channel,
        should_use_precomputed=True,
    )

print("+++ Registered:", sequence, sep="\n")
if prmt.show_registered_channels:
    qvwr.ShowSequence(sequence)


LOGGER.info("--- ADD COMPUTED CHANNEL(S)")
sequence.AddComputedChannel("POL optimized", RescaledPOLChannel)

for should_add, ratio_channel in zip(prmt.add_ratio_channels, prmt.ratio_channels):
    if should_add:
        ratio_name = f"{ratio_channel[0]}_over_{ratio_channel[1]}"
        print("    " + ratio_name)
        RatioChannel = lambda _chl: _chl[ratio_channel[0]] / (
            _chl[ratio_channel[1]] + 0.1
        )
        sequence.AddComputedChannel(ratio_name, RatioChannel)

print("+++ With Computed Channel(s):", sequence, sep="\n")
if prmt.show_computed_channels:
    qvwr.ShowSequence(sequence)


LOGGER.info(f"--- CELL SEGMENTATION on {prmt.segmentation_channel}")
if "segmentations" in cache:
    segmentations = cache["segmentations"]
    requested_length = prmt.last_frame - prmt.first_frame + 1
    if (actual := segmentations.__len__()) < requested_length:
        LOGGER.error(
            f"Segmentations in cache contain too few frames. "
            f"Actual={actual}. Expected={requested_length}"
        )
        sstm.exit(-1)
    segmentations = segmentations[:requested_length]
    segmentations = segmentations_t.NewFromDicts(segmentations)
else:
    frames = sequence.Frames(channel=prmt.segmentation_channel)
    if prmt.should_normalize_contrast_before_segmenting and (
        prmt.normalization_percent_range is not None
    ):
        PreProcessed = lambda _frm: ContrastNormalized(
            _frm, prmt.normalization_percent_range
        )
    else:
        PreProcessed = None
    PostProcessed = lambda _frm: prss.WithSmallObjectsAndHolesRemoved(
        _frm, prmt.min_cell_area, prmt.max_cell_hole_area
    )
    segmentations = sgmt.SegmentationsWithTFNetwork(
        frames,
        prmt.cell_segmentation_network,
        threshold=prmt.segmentation_network_threshold,
        PreProcessed=PreProcessed,
        PostProcessed=PostProcessed,
    )
    segmentations = segmentations_t.NewFromCompartmentSequences(
        sequence.length, cells_sgms=segmentations
    )
    segmentations.CorrectBasedOnTemporalCoherence(
        min_jaccard=prmt.min_jaccard_for_fused_cell_splitting,
        max_area_discrepancy=prmt.max_area_discrepancy_for_fused_cell_splitting,
    )
    segmentations.ClearBorderObjects()
    cache["segmentations"] = segmentations.AsDicts()

cche.CloseCache(cache)
del cache

print(f"+++ Segmentation versions: {segmentations.available_versions}")
if prmt.show_segmentation:
    gvwr.ShowSegmentation(segmentations)

sequence.AddCellsFromSegmentations(prmt.segmentation_channel, segmentations)

if prmt.should_test_show_sequence:
    # Keep figure assignments independent (no "_ = ..."'s for example) to maintain figures references
    _1 = qvwr.ShowSequence(
        sequence,
        mode="tunnels",
        keep_every=3,
        prepare_only=True,
        dbe=DRAWING_CONTEXT_VEDO,
    )
    _2 = qvwr.ShowSequence(
        sequence,
        mode="mille-feuille",
        keep_every=3,
        prepare_only=True,
        dbe=DRAWING_CONTEXT_VEDO,
    )
    _3 = qvwr.ShowSequence(sequence, prepare_only=True)
    v_figure_t.ShowAll(in_main_thread=False)
    m_figure_t.ShowAll(in_main_thread=False)

print(f"+++ Cells on first frame: {sequence.cell_frames[0].cells.__len__()}")


LOGGER.info("--- FEATURES")
IntensityVarianceInCell = lambda _cll, _frm: ftre.NumpyArrayScalarFeatureInCell(
    _cll, _frm, nmpy.var
)
IntensityEntropyInCell = lambda _cll, _frm: ftre.IntensityEntropyInCell(
    _cll, _frm, n_bins=prmt.n_bins_for_entropy_computation
)
IntensityAverageInCell = lambda _cll, _frm: ftre.NumpyArrayScalarFeatureInCell(
    _cll, _frm, nmpy.mean
)
features = (
    ("POL variance", IntensityVarianceInCell, "POL optimized"),
    ("POL entropy", IntensityEntropyInCell, "POL optimized"),
    ("POL optimized", IntensityAverageInCell, "POL optimized"),
    ("cfp_over_yfp", IntensityAverageInCell, "cfp_over_yfp"),
)
for name, Function, channel in features:
    sequence.AddRadiometricFeature(name, Function, channel)


LOGGER.info("--- CELL TRACKING")
tracker = tracker_t(
    shape=sequence.shape,
    min_jaccard=prmt.min_jaccard_for_tracking,
    division_min_jaccard=prmt.min_jaccard_for_cell_division,
)
tracks, invalid_tracks = trck.CellsTracks(
    sequence.length, sequence.cells_iterator, tracker
)
sequence.AddTracks(tracks, invalid_tracks)

sequence.PrintValidInvalidSummary()
if prmt.show_tracking:
    # Keep figure assignments independent (no "_ = ..."'s for example) to maintain figures references
    _1 = tvwr.ShowUnstructuredTracking3D(
        sequence, with_cell_labels=False, prepare_only=True
    )
    _2 = tvwr.ShowTracking3D(
        sequence, with_cell_labels=False, prepare_only=True
    )
    _3 = tvwr.ShowTracking2D(sequence, prepare_only=True)
    v_figure_t.ShowAll(in_main_thread=False)
    m_figure_t.ShowAll()


LOGGER.info("--- ANALYSIS")
dividing_cells = sequence.tracks.DividingCells(with_time_point=True)
cell_division_frame_idc = {}
cell_death_response_idc = {}
cell_death_frame_idc = {}
available_features = sequence.available_cell_features
for feature, filter_, threshold in zip(
    ("variance", "entropy"),
    (death.FilterForVariance(), death.FilterForEntropy()),
    (prmt.death_threshold_for_variance, prmt.death_threshold_for_entropy),
):
    cell_death_response_idc[feature] = {}
    cell_death_frame_idc[feature] = {}

    event_feature = f"POL {feature}"
    if event_feature not in available_features:
        wrng.warn(f"{event_feature}: Missing event-related feature")
        continue

    evolutions = sequence.FeatureEvolutionsAlongAllTracks(event_feature)
    for label, (track, evolution) in evolutions.items():
        if label in cell_division_frame_idc:
            division_time_points = cell_division_frame_idc[label]
            if division_time_points == (0,):
                last_div_frm = 0
            else:
                last_div_frm = division_time_points[-1] + 1
        else:
            division_time_points = sorted(
                _elm[1] for _elm in dividing_cells if _elm[0] in track
            )
            if division_time_points.__len__() > 0:
                last_div_frm = division_time_points[-1] + 1
            else:
                division_time_points = (0,)
                last_div_frm = 0
            cell_division_frame_idc[label] = division_time_points

        if evolution[0] is not None:
            death_frame_idx, death_response = death.PredictedDeathTime(
                evolution,
                5,
                evolution.__len__() - 5 - 1,
                threshold,
                last_div_frm,
                filter_,
            )
            if death_frame_idx is None:
                death_frame_idx = -1
            cell_death_response_idc[feature][label] = death_response
            cell_death_frame_idc[feature][label] = death_frame_idx


LOGGER.info("--- SAVING RESULTS")
ftst.SaveCellFeatureToXLSX(prmt.file_for_cell_features, sequence)
ftst.SaveCellEventsToXLSX(
    prmt.file_for_cell_events,
    cell_division_frame_idc,
    cell_death_frame_idc,
    death_response=cell_death_response_idc,
)
sqio.SaveAnnotatedSequence(sequence, prmt.file_for_frames_w_track_labels)
