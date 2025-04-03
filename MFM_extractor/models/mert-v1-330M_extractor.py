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

class MertV1330MExtractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the MERT-v1-330M extractor using m-a-p/MERT-v1-330M.
        
        In this version, we use a single aggregator approach:
        a Conv1d that merges the 25 hidden layers into one embedding.
        
        :param device: Device to run the model on ('cpu' or 'cuda').
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        print(self.device)
        
        # Load the model and processor using the 330M weights
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

        # Aggregator for 25 layers: in_channels=25, out_channels=1
        self.aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1).to(self.device)

        # Use the processor's sampling rate if available; default to 16000 otherwise
        self.resample_rate = getattr(self.processor, "sampling_rate", 16000)
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract_features(self, audio_path):
        """
        Extract features from a single .wav file using MERT-v1-330M.
        Returns a numpy array of aggregated embeddings.
        (This method remains for individual processing.)
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

        # Process the audio input with the feature extractor
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.resample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Extract features with hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Stack all hidden states: expected shape (num_layers, batch=1, time, hidden_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)  # shape: (25, time, hidden_dim)
        
        # Average over the time dimension -> shape: (25, hidden_dim)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=-2)
        
        # Prepare for aggregator: add batch dimension -> (1, 25, hidden_dim)
        time_reduced_hidden_states = time_reduced_hidden_states.unsqueeze(0)
        
        # Apply aggregator: output shape (1, 1, hidden_dim); squeeze to (hidden_dim,)
        aggregated = self.aggregator(time_reduced_hidden_states)
        aggregated = aggregated.squeeze(0).squeeze(0)
        
        return aggregated.detach().cpu().numpy()

    def extract_features_batch(self, batch_audio_arrays):
        """
        Process a batch of audio files.
        
        :param batch_audio_arrays: List of 1D numpy arrays, each representing an audio file.
        :return: List of aggregated embeddings (numpy arrays) for each file.
        """
        fs = self.resample_rate
        # Process the batch with padding using the processor.
        inputs = self.processor(batch_audio_arrays, sampling_rate=fs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # outputs.hidden_states is a tuple of 25 tensors, each of shape (batch, time, hidden_dim)
        # Stack them to obtain shape: (25, batch, time, hidden_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        # Average over the time dimension (dim=2) -> shape: (25, batch, hidden_dim)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=2)
        # Permute to shape: (batch, 25, hidden_dim)
        time_reduced_hidden_states = time_reduced_hidden_states.permute(1, 0, 2)
        # Pass through aggregator: expects shape (batch, in_channels, seq_len) where in_channels=25
        aggregated = self.aggregator(time_reduced_hidden_states)  # shape: (batch, 1, hidden_dim)
        aggregated = aggregated.squeeze(1)  # shape: (batch, hidden_dim)
        return [emb.detach().cpu().numpy() for emb in aggregated]


    def extract_folder(self, folder_path, output_file):
        """
        Process all .wav files in the folder and save extracted features to a CSV file.
        Uses batch and (optionally) parallel processing.
        """
        data_records = []
        filenames = []
        list_of_batches = []
        batch_audio_arrays = []
        batch_names = []
        
        # Get sorted list of audio files for consistent ordering.
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
            # Convert waveform to a 1D numpy array.
            audio_array = waveform.squeeze().numpy().astype(np.float32)
            batch_audio_arrays.append(audio_array)
            batch_names.append(filename)
            if len(batch_audio_arrays) == self.batch_size:
                list_of_batches.append((batch_audio_arrays.copy(), batch_names.copy()))
                batch_audio_arrays = []
                batch_names = []
        if batch_audio_arrays:
            list_of_batches.append((batch_audio_arrays.copy(), batch_names.copy()))
        
        # Process batches either sequentially or in parallel.
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
