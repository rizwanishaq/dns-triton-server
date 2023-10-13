import grpc
import pyaudio
from six.moves import queue
from tritonclient.utils import *
import tritonclient.grpc as grpcclient
from uuid import uuid4
from utils import unique_sequence_id
import timeit
from scipy.io import wavfile

# Audio recording parameters
RATE = 8000
CHUNK = int(RATE * (20/1000))  # 20ms


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._audio_interface = pyaudio.PyAudio()
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Split chunk if it contains more than 320 samples
            while len(chunk) > self._chunk:
                data.append(chunk[:self._chunk])
                chunk = chunk[self._chunk:]

            yield b''.join(data)


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
    def __init__(self, rate):
        self._rate = rate
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = None

    def __enter__(self):
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            output=True,
            frames_per_buffer=2048
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
        with AudioDenoiserStream(model_name="denoiser", triton_url="0.0.0.0:8001") as audio_stream, \
                MicrophoneStream(RATE, CHUNK) as stream, AudioPlayer(RATE) as audio_player:
            audio_generator = stream.generator()

            for content in audio_generator:
                audio = np.frombuffer(content, dtype=np.int16)
                audio = audio[:160]
                start_time = timeit.default_timer()  # Start the timer
                try:
                    response = audio_stream(audio)
                    audio_player.play(np.array(response, dtype=np.int16))

                    end_time = timeit.default_timer()  # End the timer
                    elapsed_time = (end_time - start_time) * \
                        1e3  # Convert to milliseconds
                    print(
                        f'Response shape: {response.shape}, Time taken: {elapsed_time:.2f} milliseconds')
                except Exception as e:
                    pass

    except KeyboardInterrupt:
        print("Keyboard Interrupt. Saving the output audio...")
    except grpc.RpcError as e:
        print(
            f"Error during gRPC communication: {e}. Saving the output audio...")
