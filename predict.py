"""
ensemble_predict.py — Chay ca 2 model (audio + R3D) tren 1 video, in ket qua.

Port tu Kaggle notebook (cell 59, ham watch_and_predict) - bo phan hien thi
video inline (Video(...) chi hoat dong trong Jupyter), thay bang CLI thuan.

LUU Y: day chi la ban DEMO SO SANH 2 du doan canh nhau, KHONG co cong thuc
hop nhat (fusion) ro rang - dung y het logic goc trong notebook. Neu can
1 quyet dinh CUOI CUNG duy nhat (vd de dua vao decision_module.py), can tu
them logic hop nhat (vd trung binh 2 xac suat, hoac uu tien model co
confidence cao hon) - chua co san trong ban goc.

CACH CHAY:
    python ensemble_predict.py --video path/to/video.mp4 --models_dir ./models
"""

import argparse
from pathlib import Path

import joblib

from features.audio_features import extract_audio_features
from features.r3d_embedding import extract_r3d_embedding


def load_models(models_dir: str):
    models_dir = Path(models_dir)
    audio_clf = joblib.load(models_dir / "fall_audio_classifier_final.pkl")
    audio_scaler = joblib.load(models_dir / "feature_scaler_final.pkl")
    r3d_clf = joblib.load(models_dir / "fall_r3d_classifier.pkl")
    r3d_scaler = joblib.load(models_dir / "r3d_feature_scaler.pkl")
    return audio_clf, audio_scaler, r3d_clf, r3d_scaler


def predict_video(video_path: str, audio_clf, audio_scaler, r3d_clf, r3d_scaler) -> dict:
    """Chay ca 2 model tren 1 video, tra ve dict ket qua tung nhanh (khong hop nhat)."""
    result = {"video_path": video_path, "audio": None, "r3d": None}

    feat_audio = extract_audio_features(video_path)
    if feat_audio is not None:
        prob_audio = audio_clf.predict_proba(audio_scaler.transform([feat_audio]))[0][1]
        result["audio"] = {"label": "FALL" if prob_audio >= 0.5 else "NON-FALL", "fall_probability": float(prob_audio)}

    feat_r3d = extract_r3d_embedding(video_path)
    if feat_r3d is not None:
        prob_r3d = r3d_clf.predict_proba(r3d_scaler.transform([feat_r3d]))[0][1]
        result["r3d"] = {"label": "FALL" if prob_r3d >= 0.5 else "NON-FALL", "fall_probability": float(prob_r3d)}

    return result


def main():
    parser = argparse.ArgumentParser(description="Du doan fall/non-fall tu 1 video bang ca 2 model (audio + R3D)")
    parser.add_argument("--video", required=True, help="Duong dan file video")
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--true_label", default=None, help="(Tuy chon) nhan that de doi chieu khi test")
    args = parser.parse_args()

    audio_clf, audio_scaler, r3d_clf, r3d_scaler = load_models(args.models_dir)

    print(f"File: {args.video}")
    if args.true_label:
        print(f"Nhan that: {args.true_label}")
    print()

    result = predict_video(args.video, audio_clf, audio_scaler, r3d_clf, r3d_scaler)

    if result["audio"]:
        a = result["audio"]
        print(f"[AUDIO]  {a['label']}  (fall prob: {a['fall_probability']:.2%})")
    else:
        print("[AUDIO]  khong trich xuat duoc (video co the khong co audio)")

    if result["r3d"]:
        r = result["r3d"]
        print(f"[R3D]    {r['label']}  (fall prob: {r['fall_probability']:.2%})")
    else:
        print("[R3D]    khong trich xuat duoc")


if __name__ == "__main__":
    main()