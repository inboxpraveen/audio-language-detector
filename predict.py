import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model as LOAD_AUDIO_LANGUAGE_DETECTION_MODEL

class audio_language_detection:

    def __init__(self, audio_model_path :str = ""):

        self.HINDI = "Hindi"
        self.ENGLISH = "English"
        self.PATH_TO_AUDIO_LANGUAGE_MODEL = audio_model_path ## needs to come through env variables.
        self.FRAME_SIZE = 0.025
        self.FRAME_STRIDE = 0.01
        self.PRE_EMPHASIS = 0.97
        self.NFB = 40
        self.NFFT = 512


    def generate_fb_and_mfcc(self, signal, sample_rate):

        # Pre-Emphasis
        pre_emphasis = self.PRE_EMPHASIS
        emphasized_signal = np.append(
            signal[0],
            signal[1:] - pre_emphasis * signal[:-1])

        # Framing
        frame_size = self.FRAME_SIZE
        frame_stride = self.FRAME_STRIDE

        # Convert from seconds to samples
        frame_length, frame_step = (
            frame_size * sample_rate,
            frame_stride * sample_rate)
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))

        # Make sure that we have at least 1 frame
        num_frames = int(
            np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))

        # Pad Signal to make sure that all frames have equal
        # number of samples without truncating any samples
        # from the original signal
        pad_signal = np.append(emphasized_signal, z)

        indices = (
            np.tile(np.arange(0, frame_length), (num_frames, 1)) +
            np.tile(
                np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1)
            ).T
        )
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # Window
        frames *= np.hamming(frame_length)

        # Fourier-Transform and Power Spectrum
        NFFT = self.NFFT

        # Magnitude of the FFT
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

        # Power Spectrum
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

        # Filter Banks
        nfilt = self.NFB

        low_freq_mel = 0

        # Convert Hz to Mel
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

        # Equally spaced in Mel scale
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

        # Convert Mel to Hz
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)

        # Numerical Stability
        filter_banks = np.where(
            filter_banks == 0,
            np.finfo(float).eps,
            filter_banks)

        # dB
        filter_banks = 20 * np.log10(filter_banks)

        return filter_banks

    
    def crop_and_make_predictions(self, audio_np):
        
        sc = StandardScaler()
        model = LOAD_AUDIO_LANGUAGE_DETECTION_MODEL(self.PATH_TO_AUDIO_LANGUAGE_MODEL)
        
        MFCC = self.generate_fb_and_mfcc(audio_np, 16000)
        MFCC_sc = sc.fit_transform(MFCC)
        MFCC_sc_array = np.array(MFCC_sc).reshape(-1,298,40,1)
        prediction = np.argmax(model.predict(MFCC_sc_array)[0])
        if prediction == 0:
            return self.ENGLISH
        else:
            return self.HINDI


def MAKE_LANGUAGE_PREDICTION(audio_np_array, model_path):
    try:
        aduio_language_detection_obj = audio_language_detection(model_path)
        predicted_language = aduio_language_detection_obj.crop_and_make_predictions(audio_np_array)
        return predicted_language
    except Exception as e:
        print("Something went wrong with prediction: ",e)
        return ""
