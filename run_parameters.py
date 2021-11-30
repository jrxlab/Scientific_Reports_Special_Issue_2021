r"""
How to specify paths on Windows:
C:\Users\me\Desktop\data ircan\movie.tiff
->
path_t("C:") / "Users" / "me" / "Desktop" / "data ircan" / "movie.tiff"
"""
from pathlib import Path as path_t


# --- Runtime
caching = False  # Should normally be False. Enable only for debugging.
should_test_show_sequence = False

# --- Sequence
sequence_number = 14
sequence_path = path_t("_data") / "sequence" / f"treat01_{sequence_number}_R3D.dv"
sequence_channels = ("cfp", "yfp", "POL")
first_frame = 0
last_frame = 9999
show_sequence = False

# --- Registration
registration_channel = "yfp"  # To correct the between-channel shifts
register_channels = True
show_registered_channels = False

# --- Additional Channels
ratio_channels = (("cfp", "yfp"),)
add_ratio_channels = (True,)
show_computed_channels = False

# --- Segmentation
segmentation_channel = "yfp"
should_normalize_contrast_before_segmenting = False
normalization_percent_range = [10, 90]
cell_segmentation_network = path_t("_data") / "hela_p53_c8_cell"
segmentation_network_threshold = 0.9
min_cell_area = 100
max_cell_hole_area = 10
min_jaccard_for_fused_cell_splitting = 0.3  # Splitting of fused cells (lower => more splittings)
max_area_discrepancy_for_fused_cell_splitting = 0.5
show_segmentation = False

# --- Features
n_bins_for_entropy_computation = 20

# --- Tracking
min_jaccard_for_tracking = 0.02 # For t<=>(t+1) cell matching
min_jaccard_for_cell_division = 0.2
show_tracking = True

# --- Analysis
death_threshold_for_variance = 0.1
death_threshold_for_entropy = 4.0

# --- Output
file_for_cell_features = "features.xlsx"
file_for_cell_events = "cell_events.xlsx"
file_for_frames_w_track_labels = "annotated_sequence.tiff"
