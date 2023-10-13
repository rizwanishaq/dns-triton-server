import grpc
import pyaudio
from six.moves import queue
from tritonclient.utils import *
import tritonclient.grpc as grpcclient
from uuid import uuid4
from utils import unique_sequence_id
import timeit
import numpy as np
from scipy.io import wavfile
import time

# Define your input and output file paths
INPUT_FILE_PATH = "input_audio4.wav"
OUTPUT_FILE_PATH = "output_audio.wav"

# Audio recording parameters
RATE = 8000
CHUNK = int(RATE * (20/1000))


class AudioDenoiserStream:
    def __init__(self, model_name="denoiser", triton_url="0.0.0.0:8001"):
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_url, verbose=False
        )
        self.model_name = model_name
        self.sequence_id = unique_sequence_id()

    def _input_output(self, audio: np.ndarray):
        inputs = [
            grpcclient.InferInput(
                "input_audio_chunk",
                audio.shape,
                np_to_triton_dtype(audio.dtype)
            ),
        ]
        outputs = [grpcclient.InferRequestedOutput("output_audio_chunk")]
        inputs[0].set_data_from_numpy(audio)
        return inputs, outputs

    def _send_request(self, audio, sequence_start=False, sequence_end=False):
        inputs, _ = self._input_output(audio)
        _ = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

    def start(self):
        # Change to zeros array of shape [320]
        empty_chunk = np.zeros(160, dtype=np.int16)
        self._send_request(empty_chunk, sequence_start=True,
                           sequence_end=False)
        print(f'[✅] Stream: {self.sequence_id} - Started')

    def close(self):
        # Change to zeros array of shape [320]
        empty_chunk = np.zeros(160, dtype=np.int16)
        self._send_request(
            empty_chunk, sequence_start=False, sequence_end=True)
        print(f'[✅] Stream: {self.sequence_id} - Closed')

    def __call__(self, audio: np.ndarray, sequence_start=False, sequence_end=False):
        inputs, outputs = self._input_output(audio)
        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=outputs,
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )
        output_audio_chunk = response.as_numpy('output_audio_chunk')[0]
        return output_audio_chunk

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class AudioPlayer:
    def __init__(self, rate=8000):
        self._rate = rate
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = None

    def __enter__(self):
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            output=True,
            frames_per_buffer=1024
        )
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def play(self, data):
        self._audio_stream.write(data.tobytes())

    def close(self):
        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        if self._audio_interface:
            self._audio_interface.terminate()


if __name__ == "__main__":
    try:
        # Load the input audio file
        rate, audio_data = wavfile.read(INPUT_FILE_PATH)
        if audio_data.ndim == 2:
            audio_data = audio_data[:, 0]
        num_frames = len(audio_data) // CHUNK

        # Initialize output audio array
        output_audio = np.zeros(0, dtype=np.int16)

        with AudioDenoiserStream(model_name="denoiser", triton_url="0.0.0.0:8001") as audio_stream, AudioPlayer(RATE) as audio_player:
            for i in range(num_frames):
                start_idx = i * CHUNK
                end_idx = (i+1) * CHUNK
                audio_frame = audio_data[start_idx:end_idx]
                try:
                    start_time = timeit.default_timer()  # Start the timer
                    response = audio_stream(audio_frame)
                    end_time = timeit.default_timer()  # End the timer
                    elapsed_time = (end_time - start_time) * \
                        1e3  # Convert to milliseconds

                    audio_player.play(np.array(response, dtype=np.int16))
                    output_audio = np.append(
                        output_audio, np.array(response, dtype=np.int16))

                    print(
                        f'Frame {i+1}/{num_frames} - Time taken: {elapsed_time:.2f} milliseconds')
                except Exception as e:
                    pass

        # Save the output audio to a file
        wavfile.write(OUTPUT_FILE_PATH, RATE, output_audio)

    except KeyboardInterrupt:
        pass
    except grpc.RpcError as e:
        pass
