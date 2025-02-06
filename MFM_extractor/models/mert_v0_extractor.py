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
    def __init__(self, device='cpu'):
        """
        Initialize the MERT-v0 extractor using m-a-p/MERT-v0.

        This extractor uses a learnable Conv1d aggregator to combine the 13 hidden layers
        into a single embedding.

        :param device: 'cpu' or 'cuda' (default 'cpu').
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')

        # Load the MERT-v0 model (with remote code) and corresponding feature extractor
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True)

        # A learnable aggregator for the 13 layers (in_channels=13, out_channels=1)
        self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(self.device)

        # Default to 16000 Hz if the processor doesn't specify a sampling_rate
        self.resample_rate = getattr(self.processor, "sampling_rate", 16000)

    def extract_features(self, audio_path):
        """
        Extract features from a single WAV file using MERT-v0.

        Steps:
        1) Load and resample audio if necessary.
        2) Run through the MERT-v0 feature extractor and model (with hidden states).
        3) Aggregate across time dimension -> shape (13, hidden_dim).
        4) Use the learnable Conv1d aggregator to produce a single embedding.
        5) Return the final embedding as a NumPy array.
        """
        try:
            waveform, sampling_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
        # Resample if sampling_rate != self.resample_rate
        if sampling_rate != self.resample_rate:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=self.resample_rate)
            waveform = resampler(waveform)

        # Process the audio input
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.resample_rate,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of length 13 => each shape (batch=1, time, hidden_dim)
        # Stack into tensor => shape (13, batch=1, time, hidden_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        # Remove the batch dimension => shape (13, time, hidden_dim)
        all_layer_hidden_states = all_layer_hidden_states.squeeze(dim=1)

        # Average over the time dimension => shape (13, hidden_dim)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=1)

        # aggregator expects shape (batch=1, in_channels=13, seq_len=hidden_dim)
        time_reduced_hidden_states = time_reduced_hidden_states.unsqueeze(0)
        # => shape: (1, 13, hidden_dim)

        # Weighted average aggregator
        weighted_avg_hidden_states = self.aggregator(time_reduced_hidden_states)
        # => shape: (1, 1, hidden_dim)

        weighted_avg_hidden_states = weighted_avg_hidden_states.squeeze(0).squeeze(0)
        # => shape: (hidden_dim,)

        return weighted_avg_hidden_states.detach().cpu().numpy()

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder and save extracted features to a CSV file.

        :param folder_path: Directory with .wav files.
        :param output_file: CSV file path for saving the extracted features.
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

        features_df = pd.DataFrame(data_records)
        features_df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        features_df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
