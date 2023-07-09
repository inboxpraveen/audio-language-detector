
def generate_fb_and_mfcc(signal, sample_rate):
    """
    The most important function to convert an input audio data into Mel Spectrogram. 
    If you do not have any background on speech recognition, check out the following references
    
    https://www.youtube.com/watch?v=hF72sY70_IQ&ab_channel=MaziarRaissi
    
    https://jonathan-hui.medium.com/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
    
    These explain the core concept behind speech recognition with deep learning.
    
    Overview of the following code in steps:
        1. Input cropped audio -> input signal & sample rate (8KHz or 16KHz)
        2. Preemphasis - increasing energy in higher frequencies that helps in suppressing background noise + increase phone detection
        3. Create matrix representation of input signal in shape (400 x 1197) -> how is this shape? 
            (400 -> each window size * frequency,
            1201 -> (((12 seconds * frequency) - 400) / step size * frequency) + 4 ## Note: 12 seconds is out cropped audio size. and + 4 = (2*2) because first 4 signal corresponds to vocal representation of signal.
            )
        4. Create discrete fourier transform of input frame. So from (400,1201) -> (number of nfft, 1201)
        5. The above is obtained in terms of complex number due to DFT containing -j in formula. So, we estimate the value in power spectrogram
            So, we divide by N and Square the value at 4th step -> (512,1201)
        6. The above numbers are in frequency. we want to work in scale in vectors rather than mathematical model.
        7. So, we will create a MEL filter bank varying between min & max values of frequency.
        8. In our case, we consider 0 as minimum and maximum we calculated based on 16Hz. so our original frequency will be represented into 0 to 2839 mel scale.
        9. The mel scale has a formula -> 2595 * ln(1+(f/2)/700) -> converting freq to mel scale -> 2839
            & mel scale inverse = 700 * (exp(m/2839) - 1) -> converting mel scale to frequency
        10. Form the triangular number of filters -> linespace(minimum mel scale,maximum mel scale, number of filter bank)
        11. This gives us (40,512) triangular shapes where 40 is number of triangles & 512 is our nfft. 
            If you get a doubt why 512 and not 2839?
            Because we are producing mel bands only on the nfft and not on min,max frequency.
        12. We now multiple triangles to corresponding energy, where energy is obtained from ith filter bank at frame j
        13. That gives our final shape from (40,512) to (40,1201)
        14. its optional to take log. when we take log, we see a log mel spectrogram, otherwise its just a mel spectrogram.


    Input: Signal -> scalar value
        CHECK "audio_language_detection\scripts\Reference_Images\1. Input Signal.png" FILE
        
    Output: Log Mel Spectrogram -> Matrix vector of shape (40,1201)
        CHECK "audio_language_detection\scripts\Reference_Images\2. Output Log Mel Spectrogram.jpg" FILE


    """
    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(
        signal[0],
        signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01

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
    NFFT = 512

    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Filter Banks
    nfilt = 40

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

    # MFCCs
    # num_ceps = 12
    # cep_lifter = 22

    # ### Keep 2-13
    # mfcc = dct(
    #     filter_banks,
    #     type=2,
    #     axis=1,
    #     norm='ortho'
    # )[:, 1 : (num_ceps + 1)]

    # (nframes, ncoeff) = mfcc.shape
    # n = np.arange(ncoeff)
    # lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    # mfcc *= lift
    #filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    
    return filter_banks