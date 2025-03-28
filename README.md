# MFM_extractor

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/CodeVault-girish/MFM-models.git
   cd  MFM-models
   ```
2. **To get list of models**
   ```
   from MFM_extractor import model_list, extract_from
   model_list()
   ```
3. **Install to avoid beckend error**   
   ```
   pip install -r requirements.txt
   ```
4. **To get the embeddings**
   ```
   extract_from(
    selection="1",
    folder_path="/path/to/your/wav/files",    # .wav
    output_file="/path/to/save/output.csv",   
    device="cuda", 
    batch_size=4, 
    num_workers=1                             
    )

