"""
train_audio_model.py — Train RandomForest tren dac trung audio (MFCC...).

Port tu Kaggle notebook (cell 21-27). Chay duoc tren may local/server -
khong con phu thuoc /kaggle/working, dung --output_dir de chi dinh noi luu.

CACH CHAY:
    python train_audio_model.py --manifest full_manifest.csv --output_dir ./models
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from features.audio_features import extract_audio_features


def extract_all_audio_features(df: pd.DataFrame, checkpoint_dir: str, checkpoint_every: int = 500):
    """Trich audio feature cho toan bo manifest, co checkpoint de phong gian doan giua chung."""
    features_list, labels_list, subject_list = [], [], []
    failed_indices = []

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio features"):
        feat = extract_audio_features(row["filepath"])
        if feat is not None:
            features_list.append(feat)
            labels_list.append(row["label_binary"])
            subject_list.append(row["subject"])
        else:
            failed_indices.append(idx)

        if (idx + 1) % checkpoint_every == 0:
            np.save(checkpoint_dir / "audio_features_checkpoint.npy", np.array(features_list))
            np.save(checkpoint_dir / "audio_labels_checkpoint.npy", np.array(labels_list))
            elapsed = time.time() - start_time
            print(f"\n[Checkpoint] {idx + 1}/{len(df)} done, {len(failed_indices)} failed, {elapsed:.0f}s elapsed")

    print(f"\nHoan tat: {len(features_list)} thanh cong, {len(failed_indices)} that bai")
    return np.array(features_list), np.array(labels_list), np.array(subject_list)


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest tren dac trung audio")
    parser.add_argument("--manifest", required=True, help="File CSV manifest (tu data/manifest.py)")
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    print(f"Da doc manifest: {len(df)} video")

    X, y, subject_list = extract_all_audio_features(df, checkpoint_dir=str(output_dir))

    if len(X) == 0:
        print("[LOI] khong trich xuat duoc dac trung nao - kiem tra lai ffmpeg da cai chua")
        return

    np.save(output_dir / "X_audio_features.npy", X)
    np.save(output_dir / "y_audio_labels.npy", y)

    # ---- Danh gia: train/test split thuong ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("\n=== Danh gia tren tap test (20%) ===")
    print(classification_report(y_test, y_pred, target_names=["non-fall", "fall"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---- Danh gia: GroupKFold theo subject (tranh ro ri du lieu) ----
    print("\n=== Danh gia GroupKFold theo subject (5 fold) ===")
    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        clf, StandardScaler().fit_transform(X), y, groups=subject_list, cv=gkf, scoring="f1"
    )
    print(f"F1 scores: {cv_scores}")
    print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # ---- Train model cuoi tren TOAN BO du lieu de deploy ----
    print("\n=== Train model cuoi (toan bo du lieu) ===")
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X)
    final_clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
    )
    final_clf.fit(X_all_scaled, y)

    joblib.dump(final_clf, output_dir / "fall_audio_classifier_final.pkl")
    joblib.dump(final_scaler, output_dir / "feature_scaler_final.pkl")
    print(f"\nDa luu model tai: {output_dir}/fall_audio_classifier_final.pkl")
    print(f"Da luu scaler tai: {output_dir}/feature_scaler_final.pkl")


if __name__ == "__main__":
    main()