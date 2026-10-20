import sys, time, soundfile as sf, torch, os
sys.path.insert(0, ".")
from voxcpm import VoxCPM
from ja_process_v2 import JapaneseTextPreprocessor, synthesize_checked, make_sensevoice_transcriber, check_reading
torch.set_num_threads(4)
model = VoxCPM.from_pretrained("models/VoxCPM2", load_denoiser=False, optimize=False, device="cpu")
tr = make_sensevoice_transcriber("asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
ref_text = "頑張れさえすれば、体力や知力がなくても、何でもできるという、一種の精神主義もある。"
text = "ここでは靴を脱いでください。"
x, sr = sf.read("jsut_out/p4_Fjsut.wav", dtype="float32")
print("file lỗi cũ (seed 2):", check_reading(text, tr(x, sr)), tr(x, sr), flush=True)
t = time.time()
wav, tries, match = synthesize_checked(model, text, os.path.abspath("ref_jsut_human.wav"), ref_text, tr, seed=2)
sf.write("audio/qa_p4.wav", wav, model.tts_model.sample_rate)
print(f"synthesize_checked: {tries} lần thử, khớp {match:.2f}, {time.time()-t:.0f}s | nghe được: {tr(wav, model.tts_model.sample_rate)}")
