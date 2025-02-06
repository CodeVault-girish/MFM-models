# MFM_extractor/extractor.py

import importlib
from .models import MODEL_REGISTRY

def model_list():
    """
    Print all available models in the MODEL_REGISTRY.
    """
    print("Available models:")
    for key, info in sorted(MODEL_REGISTRY.items(), key=lambda x: int(x[0])):
        print(f"{key}. {info['name']}")

def extract_from(selection, folder_path, output_file, device='cpu', combine_mode='weighted'):
    """
    Launch feature extraction for the selected model.
    """

    if selection not in MODEL_REGISTRY:
        print(f"Invalid selection '{selection}'.")
        model_list()
        return

    model_info = MODEL_REGISTRY[selection]
    # If you are directly storing "extractor_class" in the registry (no lazy import):
    extractor_class = model_info["extractor_class"]

    # Instantiate the extractor with device, combine_mode if the class accepts them
    try:
        extractor = extractor_class(device=device, combine_mode=combine_mode)
    except TypeError:
        # Some classes might not accept these arguments
        extractor = extractor_class()

    # Run extraction
    extractor.extract_folder(folder_path, output_file)
