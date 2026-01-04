from model.LaBraM.preprocessing import LaBraMProcessing

class NeuroRVQProcessing(LaBraMProcessing):
    """
    Preprocessing pipeline for NeuroRVQ.
    Inherits directly from LaBraMProcessing as they share the same standards:
    - 200 Hz Resampling
    - Bandpass Filtering
    - Normalization (NeuroRVQ paper implies similar standard)
    """
    pass
