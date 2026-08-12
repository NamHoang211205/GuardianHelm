"""
interfaces.py
--------------
HOP DONG GIAO TIEP giua 4 module cua HelmetGuard.

MUC DICH: Moi nhom nho code RIENG BIET (du la tren Kaggle Notebook, nhanh Git
rieng, hay may ca nhan) DEU PHAI tra ve dung cac dataclass duoc dinh nghia o
day. Track 6 (Integration) chi viec import va noi cac ham lai voi nhau theo
dung luong da thiet ke, khong can viet lai code cua ai ca.

QUY UOC: moi nguoi implement dung CHU KY HAM (function signature) o duoi,
noi dung ben trong ham muon xu ly the nao cung duoc, mien la input/output
dung dinh dang.

Truoc deadline tich hop (cuoi Ngay 1), moi track dong goi code cua minh
thanh 1 file .py rieng (vi du: cv_module.py, audio_module.py,
decision_module.py) va dam bao co it nhat 1 ham khop dung interface duoi day.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ============================================================
# MODULE 1 — CV (Track 1: optical flow + horizon detection)
# ============================================================

@dataclass
class CVResult:
    """Output cua module CV sau khi phan tich 1 cua so video (vi du 2-3 giay)."""
    anomaly_score: float          # 0.0 (binh thuong) -> 1.0 (rat bat thuong)
    is_suspect: bool              # True neu anomaly_score vuot nguong, dung de trigger Module 3
    signal_source: str            # "optical_flow" | "horizon" | "combined" - de debug biet tin hieu nao trigger
    timestamp: float              # unix timestamp cua cua so video vua xu ly
    debug_info: Optional[dict] = field(default_factory=dict)  # optional, cho phep nhet them so lieu debug (vd optical flow magnitude trung binh)


def detect_anomaly(frames: list[np.ndarray], fps: float = 30.0) -> CVResult:
    """
    INPUT của Track 1 (CV): 1 danh sach frame (cua so truot, vi du 60-90 frame
    cho 2-3 giay o 30fps), moi frame la np.ndarray dang BGR (giong OpenCV doc ra).

    OUTPUT: 1 CVResult duy nhat cho ca cua so do (khong phai tung frame rieng).

    Track 1 tu quyet dinh ben trong ham nay dung optical flow, horizon
    detection, hay ket hop ca 2 - Track 6 khong quan tam, chi can dung
    signature va dung kieu du lieu tra ve.
    """
    raise NotImplementedError("Track 1 (CV) implement ham nay")


# ============================================================
# MODULE 3 — Audio (Track 3: TTS hoi tham + ASR/keyword phan loai)
# ============================================================

class ResponseType(str, Enum):
    SAFE = "safe"                # nguoi dung xac nhan an toan
    DISTRESS = "distress"        # nguoi dung cau cuu / dau don
    NO_RESPONSE = "no_response"  # im lang, khong phan hoi trong thoi gian cho


@dataclass
class AudioResult:
    """Output cua module Audio sau khi hoi tham va cho phan hoi."""
    response_type: ResponseType
    transcript: Optional[str] = None   # noi dung ASR nhan dien duoc (rong neu NO_RESPONSE)
    confidence: float = 0.0            # do tin cay cua ASR/keyword-matching, 0.0-1.0
    wait_duration_sec: float = 0.0     # thuc te da cho bao lau moi co ket qua nay


def ask_and_classify(trigger_info: CVResult, timeout_sec: float = 8.0) -> AudioResult:
    """
    INPUT: CVResult tu Module 1 (dung de biet ly do trigger, co the dung de
    log/debug), va thoi gian cho toi da truoc khi coi la NO_RESPONSE.

    Track 3 tu lo phan phat cau hoi qua TTS, ghi am, chay ASR (khuyen nghi
    PhoWhisper), roi so khop tu khoa de phan loai ra 1 trong 3 ResponseType.

    OUTPUT: 1 AudioResult duy nhat sau khi ket thuc qua trinh hoi-dap.
    """
    raise NotImplementedError("Track 3 (Audio) implement ham nay")


# ============================================================
# MODULE 4 — Decision & Alert (Track 4: rule-based fusion + escalate)
# ============================================================

class ActionType(str, Enum):
    CANCEL = "cancel"      # huy canh bao, quay lai trang thai binh thuong
    ESCALATE = "escalate"  # kich hoat gui canh bao khan cap


@dataclass
class DecisionResult:
    """Output cuoi cung, dung de Track 5 (Hardware) hoac module gui SMS/GPS su dung."""
    action: ActionType
    priority: int              # 1 (thap, vd im lang keo dai) -> 3 (cao nhat, vd cau cuu ro rang)
    reason: str                 # ly do bang text, dung de log/hien thi debug khi demo
    cv_result: CVResult = None       # giu lai ket qua goc de audit/debug
    audio_result: AudioResult = None # giu lai ket qua goc de audit/debug


def decide_action(cv_result: CVResult, audio_result: AudioResult) -> DecisionResult:
    """
    INPUT: ket qua tu Module 1 va Module 3.

    Logic mac dinh de tham khao (Track 4 co the tinh chinh them):
      - audio_result.response_type == SAFE        -> CANCEL
      - audio_result.response_type == DISTRESS     -> ESCALATE, priority=3
      - audio_result.response_type == NO_RESPONSE  -> ESCALATE, priority=2

    OUTPUT: 1 DecisionResult, day la diem cuoi cung truoc khi he thong hanh dong that.
    """
    raise NotImplementedError("Track 4 (Decision) implement ham nay")


@dataclass
class AlertPayload:
    """Du lieu can co de gui canh bao that (SMS/goi dien kem GPS)."""
    latitude: float
    longitude: float
    priority: int
    message: str
    contact_list: list[str] = field(default_factory=list)  # so dien thoai nguoi than/cuu ho


def send_alert(decision: DecisionResult, gps: tuple[float, float]) -> bool:
    """
    INPUT: DecisionResult (chi goi ham nay neu decision.action == ESCALATE)
    va toa do GPS hien tai (latitude, longitude).

    Track 4 implement goi Twilio API (hoac log console de demo) o day.
    Neu mat mang, luu vao SQLite queue va return False, co co che retry rieng.

    OUTPUT: True neu gui thanh cong, False neu that bai (se duoc retry sau).
    """
    raise NotImplementedError("Track 4 (Decision/Alert) implement ham nay")


# ============================================================
# VI DU CACH TRACK 6 (INTEGRATION) NOI 4 MODULE LAI VOI NHAU
# ============================================================

def run_pipeline_once(frames: list[np.ndarray], gps: tuple[float, float]) -> Optional[DecisionResult]:
    """
    Vi du luong xu ly 1 lan (Track 6 se goi ham nay lien tuc theo vong lap
    doc camera thuc te). Day CHI la vi du minh hoa cach ghep - khong can
    sua gi trong ham nay, chi can 4 ham tren duoc implement dung interface
    la pipeline nay chay duoc ngay.
    """
    cv_result = detect_anomaly(frames)

    if not cv_result.is_suspect:
        return None  # binh thuong, khong lam gi ca, tiep tuc vong lap doc camera

    audio_result = ask_and_classify(cv_result)
    decision = decide_action(cv_result, audio_result)

    if decision.action == ActionType.ESCALATE:
        send_alert(decision, gps)

    return decision


if __name__ == "__main__":
    # Demo chay thu voi frame gia (khong phai video that) - chi de test xem
    # 4 ham co dung interface hay chua truoc khi ghep that.
    dummy_frames = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(60)]
    dummy_gps = (21.0285, 105.8542)  # Ha Noi, vi du
    try:
        result = run_pipeline_once(dummy_frames, dummy_gps)
        print("Pipeline chay thanh cong:", result)
    except NotImplementedError as e:
        print(f"Chua implement day du: {e}")