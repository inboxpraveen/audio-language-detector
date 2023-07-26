import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from tensorflow.keras.models import load_model as LOAD_AUDIO_LANGUAGE_DETECTION_MODEL
import time


class audio_language_detection:

    def __init__(self,audio_file_path :str ="",audio_duration : float=0, audio_model_path :str = ""):
        self.FILE_EXTENSION = ""
        self.AUDIO_FILE_NAME = ""
        if audio_file_path:
            self.AUDIO_FILE_PATH = audio_file_path
            self.FILE_EXTENSION = audio_file_path.split(".")[-1]
            self.AUDIO_FILE_NAME = audio_file_path.split("/")[-1].split(".")[0]
        else:
            self.AUDIO_FILE_PATH = "NO AUDIO PATH FOUND"

        self.AUDIO_DURATION = audio_duration
        self.TEMPORARY_OUTPUT_PATH = "/tmp/"
        self.AUDIO_SEGMENT_SIZE = 3 ## seconds. *DO NOT CHANGE IT*
        self.MINIMUM_EXPECTED_AUDIO_LENGTH = 3 ## seconds
        self.MINIMUM_AUDIO_SEGMENTS = 1 ## windows
        self.TIMEOUT = 3 ## seconds to wait before returning error and trying to checking if file exists
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

    
    def crop_and_make_predictions(self):

        result_dictionary = {
            self.ENGLISH : 0,
            self.HINDI : 0
        }

        sc = StandardScaler()
        model = LOAD_AUDIO_LANGUAGE_DETECTION_MODEL(self.PATH_TO_AUDIO_LANGUAGE_MODEL)
        print(f"====== LOADED AUDIO LANGUAGE DETECTION MODEL ========")

        if self.AUDIO_DURATION > self.MINIMUM_EXPECTED_AUDIO_LENGTH:

            if self.AUDIO_FILE_NAME != "NO AUDIO PATH FOUND" and self.FILE_EXTENSION == "wav":
                ## then we can form 4 segments since 60 < // 12 >= 5..
                for multiplier in range(self.MINIMUM_AUDIO_SEGMENTS):
                    starttime_with_multiplier = multiplier * self.AUDIO_SEGMENT_SIZE
                    endtime_with_multiplier = (multiplier + 1) * self.AUDIO_SEGMENT_SIZE
                    new_filepath = self.TEMPORARY_OUTPUT_PATH + "{0}_{1}_{2}.wav".format(
                        self.AUDIO_FILE_NAME,starttime_with_multiplier,endtime_with_multiplier
                    )
                    try:
                        os.popen("ffmpeg -ss {0} -i {1} -t {2} -c copy {3} -y".format(
                            starttime_with_multiplier,self.AUDIO_FILE_PATH,self.AUDIO_SEGMENT_SIZE,new_filepath
                        )).read()
                    except:
                        ## should not proceed and return error as we are not able to crop audio so model will not be able to make predictions
                        print(f"====== SOMETHING WENT WRONG WHILE CREATING 12 SECONDS CHUNKS ========")
                        return "ERROR"

                    ## make predictions on the cropped file
                    if os.path.isfile(new_filepath):
                        # if the cropped file is successfully formed, then make predictions and update the dictionary
                        flac, samplerate = sf.read(new_filepath)
                        MFCC = self.generate_fb_and_mfcc(flac, samplerate)
                        MFCC_sc = sc.fit_transform(MFCC)
                        try:
                            MFCC_sc_array = np.array(MFCC_sc).reshape(-1,298,40,1)
                            prediction = np.argmax(model.predict(MFCC_sc_array)[0])
                            if prediction == 0:
                                result_dictionary[self.ENGLISH] = result_dictionary.get(self.ENGLISH,0) + 1
                            else:
                                result_dictionary[self.HINDI] = result_dictionary.get(self.HINDI,0) + 1
                            os.popen("rm -rf {0}".format(new_filepath))
                            time.sleep(0.5)
                        except Exception as E:
                            print(f"====== SKIPPING CURRENT SEGMENT DUE TO THE ERROR: {str(E)} ========")
                            pass
                    else:
                        while not os.path.isfile(new_filepath):
                            self.TIMEOUT -= 1
                            time.sleep(1)
                            if self.TIMEOUT == 0: ## so that we dont fall in endless loop
                                self.TIMEOUT = 10 ## restore default back original value before breaking
                                break
                        if os.path.isfile(new_filepath):
                            # if the cropped file is successfully formed, then make predictions and update the dictionary
                            flac, samplerate = sf.read(new_filepath)
                            MFCC = self.generate_fb_and_mfcc(flac, samplerate)
                            MFCC_sc = sc.fit_transform(MFCC)
                            try:
                                MFCC_sc_array = np.array(MFCC_sc).reshape(-1,1201,40,1)
                                prediction = np.argmax(model.predict(MFCC_sc_array)[0])
                                if prediction == 0:
                                    result_dictionary[self.ENGLISH] = result_dictionary.get(self.ENGLISH,0) + 1
                                else:
                                    result_dictionary[self.HINDI] = result_dictionary.get(self.HINDI,0) + 1
                                os.popen("rm -rf {0}".format(new_filepath))
                                time.sleep(0.5)
                            except Exception as E:
                                print(f"====== SKIPPING CURRENT SEGMENT DUE TO THE ERROR: {str(E)} ========")
                                pass
                print(f"====== FINAL PREDICTION FOR THE AUDIO {self.AUDIO_FILE_NAME} is : {str(result_dictionary)} ========")
                return [KEY for KEY in result_dictionary.keys() if result_dictionary[KEY] == max(result_dictionary.values())][0]
        else:
            try:
                total_number_of_segments_possible = int(self.AUDIO_DURATION // self.AUDIO_SEGMENT_SIZE)
            except Exception as E:
                ## taking default number of segments if any divide by exception occurs
                total_number_of_segments_possible = 0
            print(f"====== AUDIO SMALLER THAN 60 SECONDS, SO USING SMALLER CHUNKS ========")
            if total_number_of_segments_possible >= 3:
                if self.AUDIO_FILE_NAME != "NO AUDIO PATH FOUND" and self.FILE_EXTENSION == "wav":
                    ## then we can form 4 segments since 60 < // 12 >= 5..
                    for multiplier in range(total_number_of_segments_possible):
                        starttime_with_multiplier = multiplier * self.AUDIO_SEGMENT_SIZE
                        endtime_with_multiplier = (multiplier + 1) * self.AUDIO_SEGMENT_SIZE
                        new_filepath = self.TEMPORARY_OUTPUT_PATH + "{0}_{1}_{2}.wav".format(
                            self.AUDIO_FILE_NAME,starttime_with_multiplier,endtime_with_multiplier
                        )
                        try:
                            os.popen("ffmpeg -ss {0} -i {1} -t {2} -c copy {3} -y".format(
                                starttime_with_multiplier,self.AUDIO_FILE_PATH,self.AUDIO_SEGMENT_SIZE,new_filepath
                            )).read()
                        except:
                            ## should not proceed and return error as we are not able to crop audio so model will not be able to make predictions
                            print(f"====== SOMETHING WENT WRONG WHILE CREATING 12 SECONDS CHUNKS ========")
                            return "ERROR"
                        ## make predictions on the cropped file
                        if os.path.isfile(new_filepath):
                            # if the cropped file is successfully formed, then make predictions and update the dictionary
                            flac, samplerate = sf.read(new_filepath)
                            MFCC = self.generate_fb_and_mfcc(flac, samplerate)
                            MFCC_sc = sc.fit_transform(MFCC)
                            try:
                                MFCC_sc_array = np.array(MFCC_sc).reshape(-1,298,40,1)
                                prediction = np.argmax(model.predict(MFCC_sc_array)[0])
                                if prediction == 0:
                                    result_dictionary[self.ENGLISH] = result_dictionary.get(self.ENGLISH,0) + 1
                                else:
                                    result_dictionary[self.HINDI] = result_dictionary.get(self.HINDI,0) + 1
                                os.popen("rm -rf {0}".format(new_filepath))
                                time.sleep(0.5)
                            except Exception as E:
                                print(f"====== SKIPPING CURRENT SEGMENT DUE TO THE ERROR: {str(E)} ========")
                                pass
                        else:
                            while not os.path.isfile(new_filepath):
                                self.TIMEOUT -= 1
                                time.sleep(1)
                                if self.TIMEOUT == 0: ## so that we dont fall in endless loop
                                    self.TIMEOUT = 10 ## restore default back original value before breaking
                                    break
                            if os.path.isfile(new_filepath):
                                # if the cropped file is successfully formed, then make predictions and update the dictionary
                                flac, samplerate = sf.read(new_filepath)
                                MFCC = self.generate_fb_and_mfcc(flac, samplerate)
                                MFCC_sc = sc.fit_transform(MFCC)
                                try:
                                    MFCC_sc_array = np.array(MFCC_sc).reshape(-1,298,40,1)
                                    prediction = np.argmax(model.predict(MFCC_sc_array)[0])
                                    if prediction == 0:
                                        result_dictionary[self.ENGLISH] = result_dictionary.get(self.ENGLISH,0) + 1
                                    else:
                                        result_dictionary[self.HINDI] = result_dictionary.get(self.HINDI,0) + 1
                                    os.popen("rm -rf {0}".format(new_filepath))
                                    time.sleep(0.5)
                                except Exception as E:
                                    print(f"====== SKIPPING CURRENT SEGMENT DUE TO THE ERROR: {str(E)} ========")
                                    pass
                    print(f"====== FINAL PREDICTION FOR THE AUDIO {self.AUDIO_FILE_NAME} is : {str(result_dictionary)} ========")
                    return [KEY for KEY in result_dictionary.keys() if result_dictionary[KEY] == max(result_dictionary.values())][0]
            else:
                print("NO REASON TO DETECT AUDIO SINCE THE AUDIO LENGTH IS TOO SMALL.")
                return "NO REASON TO DETECT AUDIO SINCE THE AUDIO LENGTH IS TOO SMALL."


def MAKE_LANGUAGE_PREDICTION(audio_wav_path, duration_of_audio_in_seconds, model_path):
    try:
        print(f"\n\n====== LOGGING AUDIO LOGS FOR THE FOLLOWING : {str(audio_wav_path)} ========")
        aduio_language_detection_obj = audio_language_detection(audio_wav_path,duration_of_audio_in_seconds,model_path)
        predicted_language = aduio_language_detection_obj.crop_and_make_predictions()
        print(f"====== FINAL RESULT FOR BOTH LANGUAGES - '{str(predicted_language)}' ========")
        if predicted_language in ["HINDI","ENGLISH"]:
            return predicted_language
        else:
            return ""
    except:
        return ""