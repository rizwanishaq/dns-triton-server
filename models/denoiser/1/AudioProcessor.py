import numpy as np
import triton_python_backend_utils as pb_utils
from typing import List, Tuple, Optional


class AudioProcessor:
    """
    Class for processing audio chunks for denoising.

    Args:
        model_name (str, optional): Name of the denoising model. Default is "denoise_waveUnet".
        chunk_size (int, optional): Size of the audio chunks. Default is 160.
        pad (int, optional): Padding size. Default is 256.
        frame_len (int, optional): Length of the audio frames. Default is 2048.

    Attributes:
        model_name (str): Name of the denoising model.
        chunk_size (int): Size of the audio chunks.
        pad (int): Padding size.
        frame_len (int): Length of the audio frames.

    Methods:
        process_audio_chunk(input_audio_chunk: np.ndarray) -> Optional[np.ndarray]:
            Process an audio chunk for denoising.

        reset_clean() -> None:
            Reset clean_bef.

        __call__(input_audio_chunk: np.ndarray) -> Optional[np.ndarray]:
            Process an audio chunk for denoising.

    """

    @staticmethod
    def map_function(s: np.ndarray, fs_orig: int, fs_target: int = 8000) -> np.ndarray:
        """
        Resample audio signal `s` to the target sample rate `fs_target`.

        Args:
            s (ndarray): Input audio signal.
            fs_orig (int): Original sample rate of the input signal.
            fs_target (int, optional): Target sample rate. Default is 8000.

        Returns:
            ndarray: Resampled audio signal.

        Raises:
            ValueError: If the input signal does not have one or two dimensions.

        """
        if s.ndim == 2:
            s = s[:, 0]
        elif s.ndim == 1 and s.dtype == np.int16:
            s = s / 0x8000
        else:
            raise ValueError(
                "Input signal must be 1D or 2D ndarray of type int16.")

        if fs_orig != fs_target:
            s = np.interp(
                np.linspace(0, len(s)-1, int(len(s)*fs_target/fs_orig)),
                np.arange(len(s)), s
            )

        return s.astype('float32')

    @staticmethod
    def to_int16(data: np.ndarray) -> np.ndarray:
        """
        Convert input array `data` to int16 format.

        Args:
            data (ndarray): Input data.

        Returns:
            ndarray: Converted data in int16 format.

        Raises:
            ValueError: If the input data type is not supported.

        """
        if data.dtype == np.float32:
            return (data * 0x7fff).astype('int16')
        elif data.dtype == np.int16:
            return data
        else:
            raise ValueError("Input data type must be float32 or int16.")

    def __init__(self, model_name: str = "denoise_waveUnet", chunk_size: int = 160, pad: int = 256, frame_len: int = 2048):
        """
        Initialize the AudioProcessor.

        Args:
            model_name (str, optional): Name of the denoising model. Default is "denoise_waveUnet".
            chunk_size (int, optional): Size of the audio chunks. Default is 160.
            pad (int, optional): Padding size. Default is 256.
            frame_len (int, optional): Length of the audio frames. Default is 2048.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.pad = pad
        self.frame_len = frame_len
        self.do_mean = True
        self.timeout = 10.0

        # Precompute weighted mean arrays for optimization
        self.weigthed_mean_f = np.append(np.zeros(1), np.arange(
            2 * self.pad - 1) + 1) / (2 * self.pad - 1)
        self.weigthed_mean_b = np.flip(self.weigthed_mean_f, axis=-1)

        # Initialize buffers and indices
        self.audio_buffer = np.array([], dtype=np.int16)
        self.del_audio_index = list(range(self.frame_len - 2 * self.pad))
        self.clean_bef = np.zeros(2 * self.pad)
        self.clean_buffer = np.array([], dtype=np.int16)
        self.del_clean_index = list(range(self.chunk_size))
        self.output_frames: List[np.ndarray] = []

    def audio_denoise(self, audio_data: np.ndarray, clean_bef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Denoise audio data and calculate cvector.

        Args:
            audio_data (np.ndarray): Input audio data.
            clean_bef (np.ndarray): Previous cleaned data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Cleaned audio data, clean_bef, and cvector.
        """

        if np.sum(np.abs(audio_data)) == 0:
            cleaned = self.map_function(audio_data, 8000)
            cvector = np.array([0.0, 1.0, 0.0])
        else:
            audio_data = np.reshape(
                self.map_function(audio_data, 8000), (1, -1))

            infer_request = pb_utils.InferenceRequest(
                model_name=self.model_name,
                requested_output_names=["clean", "cvector"],
                inputs=[pb_utils.Tensor('audio', audio_data)]
            )

            infer_response = infer_request.exec()

            cleaned = pb_utils.get_output_tensor_by_name(
                infer_response, "clean").as_numpy()[0]

            cvector = pb_utils.get_output_tensor_by_name(
                infer_response, "cvector").as_numpy()[0]

        if self.do_mean:
            result = cleaned[:-2 * self.pad]
            result[:2 * self.pad] = result[:2 * self.pad] * \
                self.weigthed_mean_f + clean_bef * self.weigthed_mean_b
            clean_bef = cleaned[-2 * self.pad:]
        else:
            result = cleaned[self.pad:-self.pad]
        return self.to_int16(result), clean_bef, cvector

    def process_audio_chunk(self, input_audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process an audio chunk for denoising.

        Args:
            input_audio_chunk (np.ndarray): Input audio chunk.

        Returns:
            Optional[np.ndarray]: Processed audio chunk, or None if no frames are available.
        """
        self.audio_buffer = np.concatenate(
            (self.audio_buffer, input_audio_chunk), axis=None)

        while len(self.audio_buffer) >= self.frame_len:

            origi_audio = np.array(
                self.audio_buffer[:self.frame_len], dtype=np.int16)
            clean_audio, self.clean_bef, cvector = self.audio_denoise(
                origi_audio, self.clean_bef)

            self.clean_buffer = np.concatenate(
                (self.clean_buffer, clean_audio), axis=None)
            while len(self.clean_buffer) >= self.chunk_size:
                frame = self.clean_buffer[:self.chunk_size]
                self.output_frames.append(frame)  # Collect the frame
                self.clean_buffer = np.delete(
                    self.clean_buffer, self.del_clean_index)
            self.audio_buffer = np.delete(
                self.audio_buffer, self.del_audio_index)

        if self.output_frames:
            return self.output_frames.pop(0)
        else:
            return None

    def reset_clean(self) -> None:
        """Reset clean_bef."""
        self.clean_bef = np.zeros(2 * self.pad)

    def __call__(self, input_audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process an audio chunk for denoising.

        Args:
            input_audio_chunk (np.ndarray): The input audio chunk.

        Returns:
            Optional[np.ndarray]: The denoised audio chunk, or None if no frames are available.
        """

        return self.process_audio_chunk(input_audio_chunk)
