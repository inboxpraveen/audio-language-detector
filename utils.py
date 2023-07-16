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


def generate_cropped_segments(input_file, segment_duration=3.0):
    # Load the input WAV file
    waveform, sample_rate = torchaudio.load(input_file)

    # Calculate the number of samples corresponding to the desired segment duration
    segment_samples = int(segment_duration * sample_rate)

    # Calculate the number of segments in the audio file
    num_segments = waveform.size(1) // segment_samples

    # Generator to yield each cropped audio segment
    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        yield (waveform[:, start_idx:end_idx], i*3, (i+1)*3)

    # If there are any remaining samples after the last full segment, yield the last segment.
    remaining_samples = waveform.size(1) % segment_samples
    if remaining_samples > 0:
        start_idx = num_segments * segment_samples
        yield (waveform[:, start_idx:], i*3, (i+1)*3)



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

