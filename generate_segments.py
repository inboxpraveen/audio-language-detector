import subprocess
import wave
import os
import math
import numpy as np

def convert_audio_to_wav(input_audio_path, output_wav_path):
    # Use Sox to convert audio to WAV with the specified settings
    command = f'sox "{input_audio_path}" -c 1 -r 16000 -b 16 "{output_wav_path}"'
    subprocess.run(command, shell=True)

def generate_segments(wav_path, segment_duration=3):
    # Open the WAV file
    with wave.open(wav_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()

        # Calculate the number of frames for a 3-second segment
        segment_frames = int(segment_duration * frame_rate)

        # Calculate the total number of segments
        num_segments = math.ceil(num_frames / segment_frames)

        for segment in range(num_segments):
            # Calculate the start and end frame of the current segment
            start_frame = segment * segment_frames
            end_frame = min((segment + 1) * segment_frames, num_frames)

            # Calculate the padding frames if the last segment is shorter than 3 seconds
            padding_frames = segment_frames - (end_frame - start_frame)

            # Read frames for the current segment
            wav_file.setpos(start_frame)
            frames = wav_file.readframes(end_frame - start_frame)
            audio_data = np.frombuffer(frames, dtype=np.int16)

            # Pad the last segment if needed
            if padding_frames > 0:
                pad_data = np.zeros(padding_frames, dtype=np.int16)
                audio_data = np.concatenate((audio_data, pad_data))

            yield audio_data


# if __name__ == "__main__":
#     input_audio_path = "input_audio.mp3"  # Replace with the path of your input audio file
#     output_wav_path = "output_audio.wav"

#     convert_audio_to_wav(input_audio_path, output_wav_path)

#     for i, segment in enumerate(generate_segments(output_wav_path)):
#         # Do whatever you want with the segments
#         print(f"Segment {i + 1}: {len(segment)} samples")
