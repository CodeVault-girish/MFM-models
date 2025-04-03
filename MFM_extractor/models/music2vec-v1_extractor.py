import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Data2VecAudioModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class Music2VecExtractor:
    def __init__(self, device='cpu', batch_size=4, num_workers=1):
        """
        Initialize the Music2Vec extractor using the m-a-p/music2vec-v1 model.
        
        This version uses a single aggregator approach: a Conv1d that merges the 13 hidden layers
        into one embedding.
        
        :param device: Device to run the model on ('cpu' or 'cuda').
        :param batch_size: Number of audio files to process in one batch.
        :param num_workers: Number of parallel workers for batch processing.
                            If 1, batches are processed sequentially.
        """
        self.device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        print(self.device)
        
        # Load the processor from facebook/data2vec-audio-base-960h
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        # Load the model weights from m-a-p/music2vec-v1
        self.model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1").to(self.device)
        
        # Aggregator for 13 layers: in_channels=13, out_channels=1
        self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(self.device)
        
        # Use the processor's sampling rate if available; default to 16000 otherwise.
        self.resample_rate = getattr(self.processor, "sampling_rate", 16000)
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract_features(self, audio_path):
        """
        Extract features from a single .wav file using the music2vec-v1 model.
        Returns a numpy array of aggregated embeddings (768-dimension).
        (This method remains for individual file processing.)
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
        
        # Reduce the time dimension by taking the mean -> shape: (13, 768)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=-2)
        
        # Add a batch dimension: shape (1, 13, 768)
        time_reduced_hidden_states = time_reduced_hidden_states.unsqueeze(0)
        
        # Apply the aggregator: output shape (1, 1, 768); squeeze to (768,)
        weighted_avg_hidden_states = self.aggregator(time_reduced_hidden_states)
        weighted_avg_hidden_states = weighted_avg_hidden_states.squeeze(0).squeeze(0)
        
        return weighted_avg_hidden_states.detach().cpu().numpy()

    def extract_features_batch(self, batch_audio_arrays):
        """
        Process a batch of audio files.
        
        :param batch_audio_arrays: List of 1D numpy arrays, each representing an audio file.
        :return: List of aggregated embeddings (numpy arrays) for each file.
        """
        fs = self.resample_rate
        # Use the processor with padding for batch processing.
        inputs = self.processor(batch_audio_arrays, sampling_rate=fs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Stack hidden states: outputs.hidden_states is a tuple of 13 tensors each with shape (batch, time, hidden_dim)
        # Stack them to shape: (13, batch, time, hidden_dim)
        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        # Average over the time dimension (dim=2) -> shape: (13, batch, hidden_dim)
        time_reduced_hidden_states = all_layer_hidden_states.mean(dim=2)
        # Permute to shape: (batch, 13, hidden_dim)
        time_reduced_hidden_states = time_reduced_hidden_states.permute(1, 0, 2)
        # Pass through aggregator: expects input shape (batch, in_channels=13, seq_len=hidden_dim)
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
        
        # Get a sorted list of .wav files for consistent ordering.
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
