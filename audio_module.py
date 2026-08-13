"""
audio_module.py — Track 3 (Audio): hoi tham bang giong noi + phan loai phan hoi.
Implement dung interface `ask_and_classify()` trong interface.py.

3 quyet dinh thiet ke chinh (khac ban dau cua teammate):

1. HOAN TOAN OFFLINE - khong goi Google STT qua mang. Ly do: kich ban chinh
   cua san pham la vung nui/deo/hem lanh - noi MAT SONG la binh thuong chu
   khong phai ngoai le. Ban cu dung Google STT, khi mat mang se fallback ve
   "OK" (huy canh bao) - dung ngay tai kich ban can escalate nhat. Dung
   PhoWhisper (transformers) chay local thay the.

2. FAIL-SAFE thay vi fail-open - bat ky truong hop khong chac chan nao
   (ASR loi, khong khop tu khoa nao, model chua load duoc) deu tra ve
   DISTRESS (escalate) thay vi SAFE. False escalate chi gay phien (nguoi
   nha/cuu ho goi lai xac nhan), false cancel co the bo lot tai nan that.

3. THU AM CO ENDPOINTING (dung som khi phat hien het noi) thay vi luon cho
   du timeout_sec - neu nguoi dung tra loi ngay giay thu 2, khong can doi
   het 8 giay moi phan loai. Giam do tre trung binh truoc khi escalate.

Cach chay thu doc lap (mo phong 1 lan trigger, dung mic that):
    python audio_module.py
"""

import os
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from interface import AudioResult, CVResult, ResponseType

# ========================= CONFIG =========================
INPUT_DEVICE_INDEX = None    # None = mic mac dinh cua he thong. Doi neu can (xem sd.query_devices())
SAMPLE_RATE = 16000

CHUNK_SEC = 0.2               # do dai 1 chunk khi thu am co endpointing
SILENCE_RMS_THRESHOLD = 0.01  # RMS duoi nguong nay coi la im lang - TUNE theo mic that
MIN_VOICE_CHUNKS = 2          # can it nhat bao nhieu chunk co tieng moi coi la "da bat dau noi"
TRAILING_SILENCE_CHUNKS = 4   # sau khi da noi, im lang lien tuc bao nhieu chunk (~0.8s) thi dung som

SAVE_DEBUG_AUDIO = True
DEBUG_DIR = "debug_recordings"

DISTRESS_PEAK_THRESHOLD = 0.35    # bien do dinh coi la "rat to" (het/ren dau)
DISTRESS_SUSTAIN_RATIO = 0.4      # ty le mau tren nguong "to" coi la keo dai (khac noi chuyen binh thuong)

ASR_MODEL_NAME = "vinai/PhoWhisper-tiny"  # doi sang -base/-small neu can chinh xac hon, danh doi latency

KEYWORDS_HELP = ["cứu", "giúp", "đau", "không ổn", "không thể", "cứu tôi", "cứu với"]
KEYWORDS_OK = ["ổn", "không sao", "bình thường", "tôi ổn", "ổn rồi"]

if SAVE_DEBUG_AUDIO and not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
if INPUT_DEVICE_INDEX is not None:
    sd.default.device = (INPUT_DEVICE_INDEX, None)


# ========================= TTS / BEEP (dung chung, decision_module co the tai su dung) =========================

_tts_engine = None


def _get_tts_engine():
    """Lazy-init - tranh crash luc import file nay o may/container khong co loa (vd CI, dev container)."""
    global _tts_engine
    if _tts_engine is None:
        import pyttsx3
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", 150)
    return _tts_engine


def speak(text: str):
    """TTS qua loa (offline, khong can mang). Khong lam crash pipeline neu khong co loa."""
    print(f"[VOICE] noi: {text}")
    try:
        engine = _get_tts_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[VOICE][WARN] TTS khong dung duoc ({e}), bo qua buoc noi")


def beep_alert(times: int = 1, freq: int = 1200, duration: float = 0.15, gap: float = 0.1):
    """Phat tieng bip canh bao (tone tu sinh, khong can file audio)."""
    print(f"[BEEP] phat {times} tieng bip...")
    try:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
        silence = np.zeros(int(SAMPLE_RATE * gap))
        sequence = []
        for _ in range(times):
            sequence.append(tone)
            sequence.append(silence)
        beep_audio = np.concatenate(sequence).astype(np.float32)
        sd.play(beep_audio, samplerate=SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"[BEEP][WARN] khong phat duoc am thanh ({e}), bo qua")


def _save_debug_audio(audio: np.ndarray):
    if not SAVE_DEBUG_AUDIO or audio.size == 0:
        return
    filename = os.path.join(DEBUG_DIR, f"rec_{datetime.now().strftime('%H%M%S')}.wav")
    write(filename, SAMPLE_RATE, audio)
    print(f"[DEBUG] da luu ban ghi: {filename}")


# ========================= THU AM CO ENDPOINTING =========================

def _record_with_endpointing(timeout_sec: float) -> tuple[np.ndarray, float, bool]:
    """
    Thu am theo chunk nho, DUNG SOM ngay khi phat hien nguoi dung da noi
    xong (im lang lien tuc sau khi co tieng), thay vi luon cho du
    timeout_sec. Neu tu dau den cuoi khong co tieng gi -> cho du timeout_sec
    roi dung (dung de biet chac la NO_RESPONSE, khong the cat som truong
    hop nay).

    Return: (audio thu duoc, thoi gian thuc te da cho (giay), co phat hien
    tieng noi hay khong).

    Neu khong mo duoc mic (thieu thiet bi - vd chay tren server test khong
    co audio hardware) -> KHONG crash, fail-safe coi nhu khong thu duoc gi
    (giong NO_RESPONSE) de decision_module xu ly tiep (escalate), thay vi
    lam sap ca pipeline.
    """
    chunk_samples = int(SAMPLE_RATE * CHUNK_SEC)
    max_chunks = max(1, int(round(timeout_sec / CHUNK_SEC)))

    chunks = []
    voiced_chunks = 0
    silence_run = 0
    started = False
    t0 = time.time()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             blocksize=chunk_samples) as stream:
            for _ in range(max_chunks):
                data, _ = stream.read(chunk_samples)
                chunk = data[:, 0]
                chunks.append(chunk)

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms >= SILENCE_RMS_THRESHOLD:
                    started = True
                    voiced_chunks += 1
                    silence_run = 0
                elif started:
                    silence_run += 1

                if started and voiced_chunks >= MIN_VOICE_CHUNKS and silence_run >= TRAILING_SILENCE_CHUNKS:
                    break  # nguoi dung da noi xong va im lang du lau - khong can cho tiep
    except Exception as e:
        print(f"[MIC][WARN] khong thu am duoc ({e}) - fail-safe coi nhu NO_RESPONSE")
        return np.zeros(0, dtype=np.float32), time.time() - t0, False

    wait_duration = time.time() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, wait_duration, started


# ========================= PHAN LOAI =========================

def detect_distress_sound(audio: np.ndarray) -> bool:
    """
    Phat hien tieng het/ren dau bang dac trung am thanh don gian (khong can
    ASR) - bien do dinh cao VA keo dai, khac voi noi chuyen binh thuong co
    nhieu khoang ngat.
    """
    if audio.size == 0:
        return False
    peak = float(np.max(np.abs(audio)))
    sustain_ratio = float(np.mean(np.abs(audio) > 0.15))
    print(f"[DISTRESS] peak={peak:.3f}, sustain_ratio={sustain_ratio:.3f}")
    return peak > DISTRESS_PEAK_THRESHOLD and sustain_ratio > DISTRESS_SUSTAIN_RATIO


_asr_pipeline = None


def _get_asr_pipeline():
    """
    Lazy-load PhoWhisper 1 lan duy nhat (nap model ton vai giay, khong the
    lam lai moi lan goi ask_and_classify). Neu khong load duoc (thieu
    dependency, khong co mang lan dau de tai checkpoint...), tra ve None -
    ben goi se fail-safe escalate thay vi crash.
    """
    global _asr_pipeline
    if _asr_pipeline is None:
        try:
            from transformers import pipeline
            _asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=-1)
        except Exception as e:
            print(f"[ASR][WARN] khong load duoc model ({e}) - se fail-safe escalate khi can ASR")
            _asr_pipeline = False  # danh dau "da thu va that bai", khong thu lai moi lan
    return _asr_pipeline or None


def _transcribe(audio: np.ndarray) -> tuple[str | None, float]:
    """
    Chay ASR offline, tra ve (text da chuan hoa lowercase, confidence).
    Confidence o day la UOC LUONG DON GIAN (khop tu khoa chinh xac -> tin
    cay cao), khong phai logprob that cua model - du dung cho muc dich
    keyword-matching cua he thong nay.
    """
    asr = _get_asr_pipeline()
    if asr is None:
        return None, 0.0
    try:
        result = asr({"array": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs={"language": "vi"})
        text = result["text"].lower().strip()
        print(f"[ASR] nhan dien: '{text}'")
        return text, 0.7
    except Exception as e:
        print(f"[ASR][WARN] loi khi transcribe ({e})")
        return None, 0.0


def _classify(audio: np.ndarray) -> tuple[ResponseType, str | None, float]:
    """Tra ve (response_type, transcript, confidence). Uu tien kiem tra re nhat truoc."""
    if detect_distress_sound(audio):
        return ResponseType.DISTRESS, None, 0.9

    text, confidence = _transcribe(audio)
    if text is None:
        # ASR khong dung duoc - khong the xac nhan an toan -> fail-safe escalate
        return ResponseType.DISTRESS, None, 0.0

    if any(k in text for k in KEYWORDS_HELP):
        return ResponseType.DISTRESS, text, max(confidence, 0.9)
    if any(k in text for k in KEYWORDS_OK):
        return ResponseType.SAFE, text, confidence

    # co noi nhung khong khop tu khoa nao - khong chac chan -> fail-safe escalate
    return ResponseType.DISTRESS, text, confidence * 0.5


# ========================= INTERFACE ENTRYPOINT =========================

def ask_and_classify(trigger_info: CVResult, timeout_sec: float = 8.0) -> AudioResult:
    """Khop dung chu ky trong interface.py. Hoi tham -> thu am -> phan loai."""
    print(f"[SYSTEM] duoc kich hoat boi CV (signal_source={trigger_info.signal_source}, "
          f"anomaly_score={trigger_info.anomaly_score:.2f})")
    beep_alert(times=1, freq=1200, duration=0.12, gap=0.0)
    speak("Bạn có ổn không?")

    audio, wait_duration, started = _record_with_endpointing(timeout_sec)
    _save_debug_audio(audio)

    if not started:
        return AudioResult(response_type=ResponseType.NO_RESPONSE, transcript=None,
                            confidence=1.0, wait_duration_sec=wait_duration)

    response_type, transcript, confidence = _classify(audio)
    print(f"[SYSTEM] ket qua phan loai: {response_type.value} (confidence={confidence:.2f})")
    return AudioResult(response_type=response_type, transcript=transcript,
                        confidence=confidence, wait_duration_sec=wait_duration)


# ========================= DEMO / TEST DOC LAP =========================

if __name__ == "__main__":
    print("San sang. Nhan ENTER de mo phong 1 lan trigger tu module CV (dung mic that).")
    dummy_trigger = CVResult(anomaly_score=0.8, is_suspect=True,
                              signal_source="manual_test", timestamp=time.time())

    while True:
        input(">> Nhan ENTER de trigger (Ctrl+C de thoat): ")
        result = ask_and_classify(dummy_trigger)
        print(f"AudioResult: {result}\n")


