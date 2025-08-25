import pandas as pd
import torch
from torchvision import datasets

def get_device(preference: str) -> str:
    """
    Return the device type as a string, honoring user preference if available.
    Falls back to 'cpu' if not available.
    """
    # Normalize input
    preference = preference.lower()

    # Get list of available backends
    available_devices = []

    if torch.cuda.is_available():
        available_devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # macOS GPU
        available_devices.append("mps")
    # You could add other checks here, like for "xpu", if you're using Intel's PyTorch extensions

    # Always fallback to cpu
    available_devices.append("cpu")

    # If user-preferred device is available, return it
    if preference in available_devices:
        return preference
    else:
        print(f"[Warning] Requested device '{preference}' not available. Using 'cpu' instead.")
        return "cpu"