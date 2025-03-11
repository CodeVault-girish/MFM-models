# MFM_extractor/models/__init__.py

MODEL_REGISTRY = {
    "1": {
        "name": "MERT-v0",
        "module": "MFM_extractor.models.mert_v0_extractor",
        "class": "MertV0Extractor"
    },
    "2": {
        "name": "MERT-v0-Public",
        "module": "MFM_extractor.models.mert-v0-public_extractor",
        "class": "MertV0PublicExtractor"
    },
    "3": {
        "name": "MERT-v1-95",
        "module": "MFM_extractor.models.mert-v1-95M_extractor",
        "class": "MertV195Extractor"
    },
    "4": {
        "name": "MERT-v1-330",
        "module": "MFM_extractor.models.mert-v1-330M_extractor",
        "class": "MertV1330MExtractor"
    }, 
    "5": {
        "name": "music2vec-v1-v1",
        "module": "MFM_extractor.models.music2vec-v1_extractor",
        "class": "Music2VecExtractor"
    } 

}
