# MFM_extractor

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/CodeVault-girish/MFM-models.git
   cd  MFM-models
   ```
   ```
   from MFM_extractor import model_list, extract_from
   model_list()
   ```
   ```
   pip install -r requirements.txt
   ```
   ```
   extract_from(
    selection="1",
    folder_path="/path/to/your/wav/files",    # .wav
    output_file="/path/to/save/output.csv",   
    device="cuda"                             
    )

