# MFM_extractor/extractor.py

import importlib
from .models import MODEL_REGISTRY

def model_list():
    print("Available models:")
    for key, info in sorted(MODEL_REGISTRY.items(), key=lambda x: int(x[0])):
        print(f"{key}. {info['name']}")

def extract_from(selection, folder_path, output_file, device='cpu', combine_mode='weighted'):
    if selection not in MODEL_REGISTRY:
        print(f"Invalid selection '{selection}'.")
        model_list()
        return

    model_info = MODEL_REGISTRY[selection]
    module_name = model_info["module"]
    class_name = model_info["class"]

    # Dynamically import the module
    mod = importlib.import_module(module_name)

    # Get the class from the module
    extractor_class = getattr(mod, class_name)

    # Instantiate the extractor
    extractor = extractor_class(device=device, combine_mode=combine_mode)
    
    # Now run extraction
    extractor.extract_folder(folder_path, output_file)
