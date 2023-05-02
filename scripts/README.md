# Export Whisper using Optimum
```
$ cd whisper
$ python3 scripts/hf_to_onnx.py -s <model size>  # model size is one of tiny, base, small, medium, large
$ python3 scripts/fp32_to_fp16.py -s <model size> -f <path to folder containing onnx models> -g
```

# Export Whisper with beam search op using ORT custom export

See the [ORT README](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/README.md) for details

# Export Whisper E2E (pre/post processing + beam search op)
Export Whisper with beam search op (see the linked README above for custom export options and optional flags):
```
$ python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-<size> --output <output folder> --use_external_data_format
```
Add pre/post processing nodes to form E2E model:
```
$ python3 whisper_e2e.py --audio <path to audio file> --model <path to whisper-<size>_beamsearch.onnx file generated from previous step>
```

# Current Status

| Description | FP32 CPU | FP32 CUDA | FP16 CUDA |
| ------------- | ------------- | ------------- | ------------- |
| HF + PT, pipeline mode | Pass | Pass | Pass |
| HF + PT, gen-and-dec mode | Pass | Pass | Pass |
| HF + PT2, pipeline mode | Pass | Pass | Pass |
| HF + PT2, gen-and-dec mode | Pass | Pass | Pass |
| HF + ORT, pipeline mode | Pass | Pass | Fail |
| HF + ORT, gen-and-dec mode | Pass | Pass | Fail |
| ORT-only, beam search op | Pass | Pass | Fail |
| ORT-only, E2E | Fail | Fail | Fail |

ORT-only, component wise:

| Description | FP32 CPU | FP32 CUDA | FP16 CUDA |
| ------------- | ------------- | ------------- | ------------- |
| Encoder | Pass | Pass | Pass |
| Decoder | Pass | Pass | Pass |
| Decoder-with-past | Pass | Pass | Pass |
| Encoder-decoder-init-for-beam-search | Pass | Pass | Pass |
| Decoder-for-beam-search | Pass | Pass | Pass |
