import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class MertV0Extractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the MERT-v0 extractor using m-a-p/MERT-v0.
        
        In this version, we use a single aggregator approach:
        a Conv1d that merges the 13 hidden layers into one embedding.
        
        :param device: Device to run the model on ('cpu' or 'cuda').
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        
        # Load the model (with remote code) and processor
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True)

        # Aggregator for 13 layers => in_channels=13, out_channels=1
        self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(self.device)

        # Use the processor's sampling rate if available; default to 16000 otherwise
        self.resample_rate = getattr(self.processor, "sampling_rate", 16000)
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract_features(self, audio_path):
        """
        Extract features from a single wav file using MERT-v0.
        Returns a numpy array of embeddings after aggregator.
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

        # Process the audio input using the processor
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.resample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Extract features with hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Stack all hidden states => shape: (num_layers, batch, time, hidden_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
        # For a single file, shape becomes (num_layers, time, hidden_dim)
 
        # Average hidden states over the time dimension (dim=1) -> (num_layers, hidden_dim)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=1)

        # Prepare for aggregator: add batch dimension => (1, num_layers, hidden_dim)
        time_reduced_hidden_states = time_reduced_hidden_states.unsqueeze(0)
        # Aggregator expects input shape (batch, in_channels, seq_len)
        aggregated = self.aggregator(time_reduced_hidden_states)  # shape: (1, 1, hidden_dim)
        aggregated = aggregated.squeeze(0).squeeze(0)  # shape: (hidden_dim,)

        return aggregated.detach().cpu().numpy()

    def extract_features_batch(self, batch_audio_arrays):
        """
        Process a batch of audio files.
        
        :param batch_audio_arrays: List of 1D numpy arrays, each representing an audio file.
        :return: List of aggregated embeddings (numpy arrays) for each file.
        """
        fs = self.resample_rate
        # Use the processor for batch processing with padding.
        inputs = self.processor(batch_audio_arrays, sampling_rate=fs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # outputs.hidden_states: tuple of tensors each with shape (batch_size, time, hidden_dim)
        # Stack them: shape becomes (num_layers, batch_size, time, hidden_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        # Average over the time dimension (dim=2) -> (num_layers, batch_size, hidden_dim)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=2)
        # Permute to (batch_size, num_layers, hidden_dim)
        time_reduced_hidden_states = time_reduced_hidden_states.permute(1, 0, 2)
        # Pass through aggregator: expects input shape (batch_size, in_channels, seq_len)
        aggregated = self.aggregator(time_reduced_hidden_states)  # shape: (batch_size, 1, hidden_dim)
        aggregated = aggregated.squeeze(1)  # shape: (batch_size, hidden_dim)
        return [emb.cpu().numpy() for emb in aggregated]

    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder and save extracted features to a CSV file.
        """
        data_records = []
        filenames = []
        list_of_batches = []
        batch_audio_arrays = []
        batch_names = []

        # Sort files for consistent ordering.
        file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".wav")])
        for filename in tqdm(file_list, desc="Processing audio files"):
            file_path = os.path.join(folder_path, filename)
            try:
                waveform, sampling_rate = torchaudio.load(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            if sampling_rate != self.resample_rate:
                resampler = T.Resample(sampling_rate, self.resample_rate)
                waveform = resampler(waveform)
            # Convert waveform to 1D numpy array.
            audio_array = waveform.squeeze().numpy().astype(np.float32)
            batch_audio_arrays.append(audio_array)
            batch_names.append(filename)
            if len(batch_audio_arrays) == self.batch_size:
                list_of_batches.append((batch_audio_arrays.copy(), batch_names.copy()))
                batch_audio_arrays = []
                batch_names = []
        if batch_audio_arrays:
            list_of_batches.append((batch_audio_arrays.copy(), batch_names.copy()))

        # Process batches: use parallel processing if num_workers > 1.
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.extract_features_batch, batch[0])
                           for batch in list_of_batches]
                for i, future in enumerate(futures):
                    batch_embeddings = future.result()
                    data_records.extend(batch_embeddings)
                    filenames.extend(list_of_batches[i][1])
        else:
            for audios, names in list_of_batches:
                batch_embeddings = self.extract_features_batch(audios)
                data_records.extend(batch_embeddings)
                filenames.extend(names)

        if not data_records:
            print("No features extracted.")
            return

        df = pd.DataFrame(data_records)
        df.insert(0, 'filename', filenames)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved all features to {output_file}")
