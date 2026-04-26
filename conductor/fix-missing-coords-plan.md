# Plan: Fix Missing Topomap Coordinates

## Objective
Provide accurate spatial coordinates for non-standard EEG channels (`CFC*`, `CCP*`, and `EOG`) in the `BCICIV` and `Inria` datasets so they can be plotted in the MNE Topomaps without errors.

## Implementation Steps

1.  **Update `generate_metadata.py`**:
    Modify the `get_mne_coords` function to include a `custom_coords` dictionary:
    *   **EOG**: Placed at the far front (`polar_angle_deg`: 0.0, `polar_radius`: 0.55).
    *   **CFC Series**: Placed midway between the Frontal-Central (FC) and Central (C) rows.
    *   **CCP Series**: Placed midway between the Central (C) and Central-Parietal (CP) rows.
    
    *If MNE's standard 10-20 lookup fails, the function will fall back to this custom dictionary.*

2.  **Re-generate Metadata**:
    Execute `python generate_metadata.py` to update the `metadata.json` files for `BCICIV_Train`, `BCICIV_Val`, `Inria_Train`, and `Inria_Val`.

3.  **Verification**:
    Run `python check_data.py --dataset BCICIV_Val --subject 1` to verify that the "Skipping channel" warnings no longer appear and the Topo Grid generates successfully.