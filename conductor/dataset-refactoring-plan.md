# Implementation Plan - Dataset and Loader Refactoring

This plan outlines the refactoring of `IO/loader.py` and `IO/dataset.py` to simplify initializers using a config-based approach, rename variables for clarity, and shift from window-based loading to patch-based processing.

## Objective
- Simplify class initializers to accept a `config` dictionary.
- Replace `window_size_to_use` with `patch_size_to_use`.
- Ensure all data is loaded from files without initial window-based cropping.
- Rename `subjects` to `subject_to_use` in config and code.
- Rename `TokenizerWrapperDataset` to `TokenizerDataset`.
- Standardize patch-length logic across all dataset wrappers.

## Proposed Changes

### 1. Configuration (`config/config.json`)
- Rename `"subjects"` to `"subject_to_use"`.
- Remove `"window_size_to_use"`.
- Add `"patch_size_to_use": 200` (or appropriate default).

### 2. Loader Refactoring (`IO/loader.py`)
- **`BaseSubjectLoader`**:
    - Update `__init__` to `(self, config: Dict, subject_id: int, desired_channel_indices: List[int])`.
    - Internalize `data_root`, `trials_to_use`, `sample_freq`, etc., from `config`.
    - Remove `window_size` and `target_points` logic from the loader level. The loader should just load the full trial.
    - Remove `pad_or_crop` or make it optional/identity if "using all data".
- **`BETALoader` & `DialLoader`**:
    - Update `__init__` signatures to match `BaseSubjectLoader`.
    - Ensure `_load_data` returns the full available time dimension for each trial.

### 3. Dataset Refactoring (`IO/dataset.py`)
- **`EEGDataset`**:
    - Update `__init__` to `(self, config: Dict, subject_list: List[int], desired_channels: List[str], loader_class: Any, transform: Optional[Callable] = None, fft_params: Optional[Dict] = None)`.
    - Pull `data_root` and `trials_to_use` from `config`.
- **Rename `TokenizerWrapperDataset` to `TokenizerDataset`**:
    - Update class name and all internal references.
    - Use `patch_size_to_use` from config if `patch_len` is not provided.
- **`MaskedPretrainDataset`**:
    - Use `patch_size_to_use` from config for patching logic.
- **`build_dataset_from_config`**:
    - Update to use new names: `subject_to_use`, `patch_size_to_use`.
    - Pass `config` directly to `EEGDataset`.

### 4. Code Cleanup & Consistency
- Update `train_tokenizer.py`, `train_pretrain.py`, `check_data.py`, `check_indices.py`, `ICA.py`, `check_neighbor.py`, and `check_reconstruction.py` to:
    - Use `dataset_params['subject_to_use']` instead of `subjects`.
    - Use `preprocess_params['patch_size_to_use']` (or `dataset_params`) instead of `window_size_to_use`.
    - Reference `TokenizerDataset` instead of `TokenizerWrapperDataset`.

## Verification Plan

### Automated Tests
- Run `check_data.py` to verify data loading still works with the new config structure.
- Run `check_indices.py` to ensure subject and label mapping is correct.
- Perform a "dry run" of `train_tokenizer.py` for 1 epoch to ensure the pipeline is intact.

### Manual Verification
- Inspect `output/visualization/data_analysis` to ensure Raw EEG and PSD plots look correct and reflect "all data" (or are appropriately handled by the viz tool).
- Verify that `patch_size_to_use` correctly influences the number of patches generated.
