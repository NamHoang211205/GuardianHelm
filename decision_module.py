"""
decision_module.py — Track 4: rule-based fusion (CV + Audio -> cancel/escalate)
va gui canh bao that (SMS/goi qua Twilio, kem GPS).

Logic decide_action() dung dung 3 rule mac dinh trong interface.py:
  SAFE -> CANCEL, DISTRESS -> ESCALATE (priority=3), NO_RESPONSE -> ESCALATE (priority=2)
Day la lop loc THU 2 sau CV - quan trong vi cv_heuristic/cv_module.py da do duoc
false-positive con cao (~54%) tren du lieu that, nen buoc hoi tham qua Audio
(SAFE -> CANCEL) chinh la co che bu lai false-positive do, khong phai chi de
xac nhan tai nan that.

send_alert(): thu gui qua Twilio truoc; neu chua cau hinh Twilio (demo/testing)
thi log console va coi nhu thanh cong; neu Twilio that bai that (mat mang...)
thi luu vao SQLite queue (alerts_queue.db) va tra ve False - retry_pending_alerts()
dung de thu gui lai khi co mang tro lai (goi dinh ky tu main.py).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from interface import ActionType, AlertPayload, AudioResult, CVResult, DecisionResult, ResponseType

DB_PATH = Path(__file__).resolve().parent / "alerts_queue.db"


# ========================= DECISION (FUSION) =========================

def decide_action(cv_result: CVResult, audio_result: AudioResult) -> DecisionResult:
    """Khop dung chu ky trong interface.py."""
    if audio_result.response_type == ResponseType.SAFE:
        decision = DecisionResult(
            action=ActionType.CANCEL,
            priority=1,
            reason=f"Nguoi dung xac nhan an toan qua giong noi "
                   f"(transcript={audio_result.transcript!r}, confidence={audio_result.confidence:.2f})",
            cv_result=cv_result,
            audio_result=audio_result,
        )
    elif audio_result.response_type == ResponseType.DISTRESS:
        decision = DecisionResult(
            action=ActionType.ESCALATE,
            priority=3,
            reason=f"Phat hien dau hieu nguy hiem qua giong noi "
                   f"(transcript={audio_result.transcript!r}, confidence={audio_result.confidence:.2f})",
            cv_result=cv_result,
            audio_result=audio_result,
        )
    else:  # NO_RESPONSE
        decision = DecisionResult(
            action=ActionType.ESCALATE,
            priority=2,
            reason=f"Khong co phan hoi sau {audio_result.wait_duration_sec:.1f}s hoi tham "
                   f"(CV trigger: {cv_result.signal_source}, anomaly_score={cv_result.anomaly_score:.2f})",
            cv_result=cv_result,
            audio_result=audio_result,
        )

    print(f"[DECISION] action={decision.action.value} priority={decision.priority} - {decision.reason}")

    # Phan hoi lai nguoi dung qua giong noi (tai su dung tien ich tu audio_module,
    # khong lam thay doi hanh vi cua decide_action ve mat du lieu tra ve).
    try:
        from audio_module import speak
        if decision.action == ActionType.CANCEL:
            speak("Đã ghi nhận, mọi thứ ổn. Chúc bạn đi đường an toàn.")
        else:
            speak("Đang gửi tín hiệu khẩn cấp và gọi cứu hộ.")
    except Exception as e:
        print(f"[DECISION][WARN] khong phat duoc thong bao giong noi ({e})")

    return decision


# ========================= GUI CANH BAO (TWILIO + SQLITE RETRY QUEUE) =========================

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            priority INTEGER NOT NULL,
            message TEXT NOT NULL,
            contact_list TEXT NOT NULL
        )
    """)
    return conn


def _queue_alert(payload: AlertPayload):
    conn = _get_db()
    conn.execute(
        "INSERT INTO pending_alerts (created_at, latitude, longitude, priority, message, contact_list) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), payload.latitude, payload.longitude,
         payload.priority, payload.message, json.dumps(payload.contact_list)),
    )
    conn.commit()
    conn.close()


def _load_contact_list() -> list[str]:
    """Doc so dien thoai nguoi than/cuu ho tu bien moi truong EMERGENCY_CONTACTS (cach nhau boi dau phay)."""
    raw = os.environ.get("EMERGENCY_CONTACTS", "")
    contacts = [c.strip() for c in raw.split(",") if c.strip()]
    if not contacts:
        print("[ALERT][WARN] chua cau hinh EMERGENCY_CONTACTS - dung so demo, "
              "dat bien moi truong nay truoc khi dung that")
        contacts = ["+84900000000"]
    return contacts


def _log_alert_console(payload: AlertPayload):
    maps_link = f"https://maps.google.com/?q={payload.latitude},{payload.longitude}"
    print("=" * 55)
    print(f"[ALERT][DEMO] priority={payload.priority}")
    print(f"  Thoi gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
    print(f"  Vi tri: {payload.latitude}, {payload.longitude} ({maps_link})")
    print(f"  Noi dung: {payload.message}")
    print(f"  Gui toi: {payload.contact_list}")
    print("=" * 55)


def _send_via_twilio(payload: AlertPayload) -> bool:
    """
    Gui SMS qua Twilio. Neu chua cau hinh (thieu bien moi truong Twilio - vd
    dang demo/dev), fallback ve log console va coi la thanh cong (dung de test
    pipeline khong can tai khoan Twilio that). Neu DA cau hinh nhung goi that
    bai (mat mang, het quota...), tra ve False de ben goi queue lai.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_FROM_NUMBER")

    if not (account_sid and auth_token and from_number):
        _log_alert_console(payload)
        return True

    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        maps_link = f"https://maps.google.com/?q={payload.latitude},{payload.longitude}"
        for to_number in payload.contact_list:
            client.messages.create(
                body=f"{payload.message} Vi tri: {maps_link}",
                from_=from_number,
                to=to_number,
            )
        return True
    except Exception as e:
        print(f"[ALERT][WARN] gui Twilio that bai ({e})")
        return False


def send_alert(decision: DecisionResult, gps: tuple[float, float]) -> bool:
    """Khop dung chu ky trong interface.py. Chi nen goi khi decision.action == ESCALATE."""
    latitude, longitude = gps
    payload = AlertPayload(
        latitude=latitude,
        longitude=longitude,
        priority=decision.priority,
        message=f"[GuardianHelm] Phat hien tai nan (priority={decision.priority}). Ly do: {decision.reason}",
        contact_list=_load_contact_list(),
    )

    if _send_via_twilio(payload):
        return True

    _queue_alert(payload)
    print("[ALERT] Gui that bai - da luu vao SQLite queue (alerts_queue.db) de retry sau.")
    return False


def retry_pending_alerts() -> int:
    """
    Thu gui lai cac alert dang cho trong queue - goi dinh ky tu main.py (vd
    moi 30s) de tu dong gui lai khi co mang tro lai sau khi mat song.
    Tra ve so alert gui thanh cong (da bi xoa khoi queue).
    """
    conn = _get_db()
    rows = conn.execute(
        "SELECT id, latitude, longitude, priority, message, contact_list FROM pending_alerts"
    ).fetchall()

    sent = 0
    for row_id, latitude, longitude, priority, message, contact_list_json in rows:
        payload = AlertPayload(
            latitude=latitude, longitude=longitude, priority=priority,
            message=message, contact_list=json.loads(contact_list_json),
        )
        if _send_via_twilio(payload):
            conn.execute("DELETE FROM pending_alerts WHERE id = ?", (row_id,))
            sent += 1

    conn.commit()
    conn.close()
    if sent:
        print(f"[ALERT] Da gui lai thanh cong {sent} alert dang cho trong queue.")
    return sent


# ========================= DEMO / TEST DOC LAP =========================

if __name__ == "__main__":
    import time

    dummy_cv = CVResult(anomaly_score=0.9, is_suspect=True, signal_source="optical_flow", timestamp=time.time())
    dummy_gps = (21.0285, 105.8542)  # Ha Noi, vi du

    print("--- Test case 1: nguoi dung xac nhan an toan ---")
    audio_safe = AudioResult(response_type=ResponseType.SAFE, transcript="tôi ổn", confidence=0.9, wait_duration_sec=2.1)
    decision = decide_action(dummy_cv, audio_safe)
    print(f"AudioResult -> {decision}\n")

    print("--- Test case 2: khong phan hoi ---")
    audio_none = AudioResult(response_type=ResponseType.NO_RESPONSE, transcript=None, confidence=1.0, wait_duration_sec=8.0)
    decision = decide_action(dummy_cv, audio_none)
    if decision.action == ActionType.ESCALATE:
        send_alert(decision, dummy_gps)
    print()

    print("--- Test case 3: nguy hiem ro rang ---")
    audio_help = AudioResult(response_type=ResponseType.DISTRESS, transcript="cứu tôi", confidence=0.9, wait_duration_sec=1.5)
    decision = decide_action(dummy_cv, audio_help)
    if decision.action == ActionType.ESCALATE:
        send_alert(decision, dummy_gps)
