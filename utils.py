import os, torchaudio

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_or_check_directory(input_directory: str = ""):
    if input_directory and isinstance(input_directory,str):
        if not os.path.exists(input_directory):
            try:
                os.makedirs(input_directory)
            except Exception as E:
                print(f"{bcolors.FAIL}[ERROR] Unable to create `{input_directory}` because of the following error: {E}{bcolors.ENDC}")
                return False
        return True
    else:
        print(f"{bcolors.FAIL}[ERROR] Unable to create `{input_directory}`. Please verify input formal parameter.{bcolors.ENDC}")
        return False

import torch
import librosa

def preprocess_audio(filepath, chunk_duration=3):
    # Load the audio file and get its sample rate
    waveform, sample_rate = torchaudio.load(filepath)

    # Convert to mono (single channel)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if the sample rate is different
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform.numpy(), orig_sr=sample_rate, target_sr=target_sample_rate)
        waveform = torch.tensor(waveform)

    # Convert to 16-bit depth
    waveform = (waveform * 32767).to(torch.int16)

    # Calculate the number of samples in a chunk
    chunk_samples = int(chunk_duration * target_sample_rate)

    # Generate chunks of 3 seconds (or less for the last chunk)
    num_chunks = waveform.size(1) // chunk_samples
    for i in range(num_chunks):
        start = i * chunk_samples
        end = (i + 1) * chunk_samples
        yield waveform[:, start:end]

    # If there's a last incomplete chunk, yield it as well
    last_chunk_start = num_chunks * chunk_samples
    if last_chunk_start < waveform.size(1):
        yield waveform[:, last_chunk_start:]



def is_empty(input_directory: str = "") -> bool:
    
    if input_directory:
        if os.path.exists(input_directory):
            if len(os.listdir(input_directory)):
                return False

    return True


def remove_files(*args) -> None:
    final_args = [x for x in args if x is not None]
    for input_directory in final_args:
        if os.path.isfile(input_directory):
            os.remove(input_directory)

