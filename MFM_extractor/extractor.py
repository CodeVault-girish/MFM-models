# MFM_extractor/extractor.py

import importlib
from .models import MODEL_REGISTRY

def model_list():
    print("Available models:")
    for key, info in sorted(MODEL_REGISTRY.items(), key=lambda x: int(x[0])):
        print(f"{key}. {info['name']}")

def extract_from(selection, folder_path, output_file, device='cpu'):
    """
    Launch feature extraction for the selected model.

    :param selection: The key (as string) of the selected model from MODEL_REGISTRY
    :param folder_path: Folder containing .wav files
    :param output_file: CSV output path
    :param device: 'cpu' or 'cuda'
    """
    if selection not in MODEL_REGISTRY:
        print(f"Invalid selection '{selection}'.")
        model_list()
        return

    model_info = MODEL_REGISTRY[selection]
    module_name = model_info["module"]
    class_name = model_info["class"]

    # Dynamically import the module
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Could not import module {module_name}: {e}")
        return

    # Retrieve the class
    try:
        extractor_class = getattr(mod, class_name)
    except AttributeError as e:
        print(f"Module '{module_name}' does not have a class named '{class_name}': {e}")
        return

    # Instantiate the extractor
    extractor = extractor_class(device=device)
    extractor.extract_folder(folder_path, output_file)
