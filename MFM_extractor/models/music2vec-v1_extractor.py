import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Data2VecAudioModel
from tqdm import tqdm

class Music2VecExtractor:
    def __init__(self, device='cpu'):
        """
        Initialize the Music2Vec extractor using the m-a-p/music2vec-v1 model.
        
        This version uses a single aggregator approach: a Conv1d that merges the 13 hidden layers
        into one embedding.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        
        # Load the processor from facebook/data2vec-audio-base-960h
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        # Load the model weights from m-a-p/music2vec-v1
        self.model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1").to(self.device)
        
        # Aggregator for 13 layers: in_channels=13, out_channels=1
        self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(self.device)
        
        # Use the processor's sampling rate if available; default to 16000 otherwise.
        self.resample_rate = getattr(self.processor, "sampling_rate", 16000)

    def extract_features(self, audio_path):
        """
        Extract features from a single .wav file using the music2vec-v1 model.
        Returns a numpy array of aggregated embeddings (768-dimension).
        """
        try:
            waveform, sampling_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
        
        # Resample the audio if needed.
        if sampling_rate != self.resample_rate:
            resampler = T.Resample(sampling_rate, self.resample_rate)
            waveform = resampler(waveform)

        # Process the audio input using the processor.
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.resample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Extract features with hidden states.
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Stack all hidden states; expected shape: (13, time, 768)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
        
        # For utterance-level tasks, reduce the time dimension by taking the mean.
        # New shape: (13, 768)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=-2)
        
        # Aggregator expects input shape: (batch_size, channels, sequence_length)
        # Unsqueeze to add a batch dimension: (1, 13, 768)
        time_reduced_hidden_states = time_reduced_hidden_states.unsqueeze(0)
        
        # Apply the learnable weighted average aggregator.
        # Output shape will be: (1, 1, 768)
        weighted_avg_hidden_states = self.aggregator(time_reduced_hidden_states)
        weighted_avg_hidden_states = weighted_avg_hidden_states.squeeze(0).squeeze(0)  # Final shape: (768,)
        
        return weighted_avg_hidden_states.detach().cpu().numpy()

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder and save extracted features to a CSV file.
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
