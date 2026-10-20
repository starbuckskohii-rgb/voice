import sys, soundfile as sf, sherpa_onnx
M = sys.argv[1]; wav = sys.argv[2]
audio, sr = sf.read(wav, dtype="float32")

def run(rec, name):
    s = rec.create_stream(); s.accept_waveform(sr, audio)
    if hasattr(rec, "decode_stream"): rec.decode_stream(s)
    print(f"{name:12}: {s.result.text}")

z = f"{M}/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
run(sherpa_onnx.OfflineRecognizer.from_transducer(encoder=f"{z}/encoder-epoch-99-avg-1.onnx", decoder=f"{z}/decoder-epoch-99-avg-1.onnx",
    joiner=f"{z}/joiner-epoch-99-avg-1.onnx", tokens=f"{z}/tokens.txt", num_threads=4), "ReazonSpeech")
v = f"{M}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
run(sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{v}/model.onnx", tokens=f"{v}/tokens.txt", language="ja", use_itn=False, num_threads=4), "SenseVoice")
w = f"{M}/sherpa-onnx-whisper-turbo"
run(sherpa_onnx.OfflineRecognizer.from_whisper(encoder=f"{w}/turbo-encoder.int8.onnx", decoder=f"{w}/turbo-decoder.int8.onnx",
    tokens=f"{w}/turbo-tokens.txt", language="ja", task="transcribe", num_threads=4), "Whisper")
