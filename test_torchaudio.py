
import torchaudio
import torch

try:
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"Backends: {torchaudio.list_audio_backends()}")
    
    wav_file = "1-100032-A-0.wav"
    print(f"Loading {wav_file}...")
    waveform, sample_rate = torchaudio.load(wav_file)
    print(f"Success! Shape: {waveform.shape}, SR: {sample_rate}")
except Exception as e:
    print(f"Failed to load: {e}")
    import traceback
    traceback.print_exc()
