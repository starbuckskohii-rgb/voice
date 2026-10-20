"""Nhận dạng âm vị (IPA) bằng wav2vec2-xlsr-53-espeak-cv-ft, giải mã CTC thủ công."""
import json, os, numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2ForCTC

HERE = os.path.dirname(os.path.abspath(__file__))
_M = os.path.join(HERE, "models/w2v2-phoneme")
_model = Wav2Vec2ForCTC.from_pretrained(_M).eval()
_vocab = {i: t for t, i in json.load(open(os.path.join(_M, "vocab.json"))).items()}
_feat = json.load(open(os.path.join(_M, "preprocessor_config.json")))

def phonemes(path_or_audio, sr=None):
    if isinstance(path_or_audio, str):
        audio, sr = sf.read(path_or_audio, dtype="float32")
    else:
        audio = path_or_audio
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    audio = np.concatenate([np.zeros(8000, np.float32), audio, np.zeros(4800, np.float32)])
    if _feat.get("do_normalize", True):
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
    with torch.inference_mode():
        ids = _model(torch.from_numpy(audio)[None]).logits[0].argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and _vocab[i] not in ("<pad>", "<s>", "</s>", "<unk>"):
            out.append(_vocab[i])
        prev = i
    return " ".join(out)
