# OpenAI Whisper with ONNX Runtime

This folder previously contained the scripts for running Whisper with ONNX Runtime (ORT). Please visit the following links for the latest code and information. 
- The [ORT README](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/README.md) contains details for exporting, optimizing, and benchmarking as well as scripts for parity verification and precision reduction.
- The [ORT Extensions example](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/whisper_e2e.py) gives you more control over how to create the pre/post processing steps and attach them to the ORT exported model in order to create an end-to-end ONNX model.
- The [Olive README](https://github.com/microsoft/Olive/tree/main/examples/whisper) contains details on how to export Whisper configurations to get an end-to-end ONNX model.

# OpenAI Whisper with Hugging Face's [Optimum](https://github.com/huggingface/optimum) + ONNX Runtime

Export:

For instructions on how to export Whisper for running in Optimum, please see [this section](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/README.md#option-3-from-hugging-face-optimum) in the ORT README.

Optimize:

For instructions on how to optimize and run the 3 `.onnx` models produced by Optimum, please see the `Usage` section in the [Whisper model optimization PR](https://github.com/microsoft/onnxruntime/pull/15473). Note that it is recommended to first optimize the FP32 model before casting down to FP16, INT8, etc.
