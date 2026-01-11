from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm


class Preprocessor:
    def __init__(self, dataset_path = Path(__file__).resolve().parent.parent / "dataset", duration=30, n_mels=128, hop_length=512):
        self.dataset_path = dataset_path
        self.duration = duration
        self.n_mels = n_mels
        self.hop_length = hop_length

    def __load_audio(self, file_path):
        y, sr = librosa.load(file_path, duration=self.duration)
        
        # Pad audio if shorter than desired duration
        target_length = sr * self.duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        return y, sr

    def __melSpectrogram(self, file_path):
        y, self.sample_rate = self.__load_audio(file_path)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def compute_melSpectrogram(self):
        x = []
        y = []
        DATASET_PATH = Path(self.dataset_path)

        n_files = sum(1 for genre_dir in DATASET_PATH.iterdir() if genre_dir.is_dir() for _ in genre_dir.glob("*.wav"))
        
        with tqdm(total=n_files, desc="Processing Dataset") as pbar:
            # Iterate over directories
            for genre_dir in DATASET_PATH.iterdir():
                if genre_dir.is_dir():
                    # Iterate over files
                    for wav_file in genre_dir.glob("*.au"):
                        mel_db = self.__melSpectrogram(wav_file)
                        x.append(mel_db)
                        y.append(genre_dir.name)
                        pbar.update(1)
        
        x = np.array(x)
        y = np.array(y)

        return x, y