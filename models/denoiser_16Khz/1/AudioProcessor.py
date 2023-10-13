import numpy as np
import triton_python_backend_utils as pb_utils


class AudioProcessor:
    """Class for processing audio chunks for denoising.

    Attributes:
        CHUNK_SIZE (int): The size of each audio chunk.
        BUFFER_SIZE (int): The size of the denoiser buffer.

    Methods:
        __init__(self):
            Initialize the AudioProcessor.

        process_audio_chunk(self, input_audio_chunk: np.ndarray) -> np.ndarray:
            Process an audio chunk for denoising.
    """

    CHUNK_SIZE: int = 320
    BUFFER_SIZE: int = 2048

    def __init__(self) -> None:
        """Initialize the AudioProcessor."""
        self.denoiser_buffer = np.zeros((self.BUFFER_SIZE,), dtype=np.float32)

    def __call__(self, input_audio_chunk: np.ndarray) -> np.ndarray:
        """Process an audio chunk for denoising.

        Args:
            input_audio_chunk (numpy.ndarray): The input audio chunk.

        Returns:
            numpy.ndarray: The denoised audio chunk.
        """
        self.denoiser_buffer = np.roll(self.denoiser_buffer, -self.CHUNK_SIZE)
        self.denoiser_buffer[-self.CHUNK_SIZE:] = input_audio_chunk

        max_abs_value = np.max(np.abs(self.denoiser_buffer))
        processed_input = np.expand_dims(
            self.denoiser_buffer / max_abs_value, axis=0)

        infer_request = pb_utils.InferenceRequest(
            model_name="denoiser16KHz",
            requested_output_names=["clean", "cvector"],
            inputs=[pb_utils.Tensor('input_1', processed_input)]
        )

        infer_response = infer_request.exec()

        clean_chunk = pb_utils.get_output_tensor_by_name(
            infer_response, "clean").as_numpy()[0]

        # Convert to int16
        clean_chunk = (clean_chunk[-self.CHUNK_SIZE:]
                       * np.iinfo(np.int16).max).astype(np.int16)

        return clean_chunk
