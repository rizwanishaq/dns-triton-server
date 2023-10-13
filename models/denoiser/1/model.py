import numpy as np
import json
import triton_python_backend_utils as pb_utils
from typing import List, Dict
from AudioProcessor import AudioProcessor


class TritonPythonModel:
    """
    Python model for Audio Denoiser.

    This class handles initialization, execution, and finalization of the model.

    Attributes:
        model_config (dict): Parsed model configuration.
        logger (pb_utils.Logger): Logger for model.
        audio_pipeline (dict): Dictionary to track audio pipelines.

    Methods:
        initialize(self, args: Dict[str, str]) -> None:
            Called once during model loading for initialization.

        execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
            Called for processing inference requests.

        finalize(self) -> None:
            Called once when the model is being unloaded.

    """

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the model.

        Args:
            args (dict): Dictionary containing model information.

                - model_config (str): JSON string containing the model configuration.
                - model_instance_kind (str): Model instance kind.
                - model_instance_device_id (str): Model instance device ID.
                - model_repository (str): Model repository path.
                - model_version (str): Model version.
                - model_name (str): Model name.
        """
        self.logger: pb_utils.Logger = pb_utils.Logger
        self.audio_pipeline: Dict = {}
        self.model_config: dict = json.loads(args['model_config'])
        output_audio_chunk_config = pb_utils.get_output_config_by_name(
            self.model_config, "output_audio_chunk")
        self.output_tensor_dtype: np.dtype = pb_utils.triton_string_to_numpy(
            output_audio_chunk_config['data_type'])

    def execute(self, requests: List) -> List:
        """
        Process inference requests.

        Args:
            requests (list): List of pb_utils.InferenceRequest.

        Returns:
            list: List of pb_utils.InferenceResponse.
        """
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            start_flag = pb_utils.get_input_tensor_by_name(
                request, "START").as_numpy()[0]
            end_flag = pb_utils.get_input_tensor_by_name(
                request, "END").as_numpy()[0]
            sequence_id = pb_utils.get_input_tensor_by_name(
                request, "CORRID").as_numpy()[0]

            audio_processor = self.audio_pipeline.get(
                sequence_id, {}).get('audio_processor')

            if end_flag or start_flag:
                if end_flag:
                    del self.audio_pipeline[sequence_id]
                    self.logger.log(
                        f'Stream {sequence_id} stopped', self.logger.INFO)
                if start_flag:
                    self.audio_pipeline[sequence_id] = {
                        "audio_processor": AudioProcessor()}
                    self.logger.log(
                        f'Stream {sequence_id} started', self.logger.INFO)

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[])
            else:
                input_audio_chunk = pb_utils.get_input_tensor_by_name(
                    request, "input_audio_chunk").as_numpy()

                output_audio_chunk = audio_processor(
                    input_audio_chunk)

                if output_audio_chunk is None:
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[])
                else:
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor(
                                "output_audio_chunk",
                                np.array(
                                    [output_audio_chunk], dtype=self.output_tensor_dtype),
                            )
                        ])

            responses.append(inference_response)

        return responses

    def finalize(self) -> None:
        """Perform cleanups before model unloading."""
        self.logger.log('Cleaning up...', self.logger.INFO)
