# MFM_extractor/models/mert_v0_extractor.py

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm

class MertV0Extractor:
    def __init__(self, device='cpu', combine_mode='weighted'):
        """
        Initialize the MERT-v0 extractor using m-a-p/MERT-v0.
        
        :param device: 'cpu' or 'cuda'
        :param combine_mode: 'weighted' or 'mean'
               - 'weighted': uses a learnable Conv1d aggregator (over 13 layers)
               - 'mean': simple average across the 13 layers
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        self.combine_mode = combine_mode
        
        # Load MERT-v0 (with remote code) and its processor
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True)

        # If aggregator is 'weighted', define a Conv1d aggregator
        # MERT typically has 13 layers => in_channels=13
        if self.combine_mode == 'weighted':
            self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(self.device)
        else:
            self.aggregator = None

        # Default to 16k if the processor doesn't specify
        self.resample_rate = getattr(self.processor, "sampling_rate", 16000)

    def extract_features(self, audio_path):
        """
        Extract features from a single .wav file.
        Returns a NumPy array of shape (feature_dim,).
        """
        try:
            waveform, sampling_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
        # Resample if needed
        if sampling_rate != self.resample_rate:
            resampler = T.Resample(sampling_rate, self.resample_rate)
            waveform = resampler(waveform)

        # Process with the MERT processor
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.resample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states => tuple of 13 layers, each (batch=1, time, hidden_dim)
        # Stack => shape (13, batch=1, time, hidden_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
        # => shape: (13, time, hidden_dim)

        # Average out time => (13, hidden_dim)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=1)

        if self.combine_mode == 'weighted':
            # Weighted aggregator => shape (batch=1, in_channels=13, length=hidden_dim)
            x = time_reduced_hidden_states.unsqueeze(0)  # => (1, 13, hidden_dim)
            weighted_avg_hidden_states = self.aggregator(x)  # => (1,1,hidden_dim)
            weighted_avg_hidden_states = weighted_avg_hidden_states.squeeze(0).squeeze(0)
            # => (hidden_dim,)
            return weighted_avg_hidden_states.detach().cpu().numpy()
        else:
            # 'mean' aggregator => average across 13 layers => shape (hidden_dim,)
            final_embedding = time_reduced_hidden_states.mean(dim=0)
            return final_embedding.detach().cpu().numpy()

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in folder_path, save embeddings to CSV at output_file.
        """
        data_records = []
        filenames = []

        for filename in tqdm(os.listdir(folder_path), desc="Processing audio files"):
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                features = self.extract_features(file_path)
                if features is not None:
                    data_records.append(features)
                    filenames.append(filename)

        if not data_records:
            print("No features extracted.")
            return

        df = pd.DataFrame(data_records)
        df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
