"""
train_r3d_model.py — Train RandomForest tren embedding trich tu R3D-18 pretrained.

Port tu Kaggle notebook (cell 52-54, 60). Chay duoc tren may local/server -
khong con phu thuoc /kaggle/working.

CACH CHAY:
    python train_r3d_model.py --manifest full_manifest.csv --output_dir ./models
    # neu chi muon chay thu nhanh tren 1 phan nho du lieu:
    python train_r3d_model.py --manifest full_manifest.csv --output_dir ./models --sample_size 600
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from features.r3d_embedding import extract_r3d_embedding


def extract_all_r3d_embeddings(df: pd.DataFrame):
    features, labels, subjects, failed = [], [], [], []
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting R3D embeddings"):
        feat = extract_r3d_embedding(row["filepath"])
        if feat is not None:
            features.append(feat)
            labels.append(row["label_binary"])
            subjects.append(row["subject"])
        else:
            failed.append(idx)

    print(f"\nHoan tat: {len(features)} thanh cong, {len(failed)} that bai")
    print(f"Thoi gian: {(time.time() - start_time) / 60:.1f} phut")
    return np.array(features), np.array(labels), np.array(subjects)


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest tren embedding R3D-18")
    parser.add_argument("--manifest", required=True, help="File CSV manifest (tu data/manifest.py)")
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--sample_size", type=int, default=None,
                         help="Neu dat, chi lay ngau nhien N video (can bang fall/non-fall) de chay nhanh thu - "
                              "R3D embedding cham hon audio nhieu, nen test tren tap nho truoc")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    print(f"Da doc manifest: {len(df)} video")

    if args.sample_size:
        n_per_class = args.sample_size // 2
        df_fall = df[df["label_binary"] == 1].sample(n=min(n_per_class, (df["label_binary"] == 1).sum()), random_state=42)
        df_nonfall = df[df["label_binary"] == 0].sample(n=min(n_per_class, (df["label_binary"] == 0).sum()), random_state=42)
        df = pd.concat([df_fall, df_nonfall]).reset_index(drop=True)
        print(f"Da lay mau: {len(df)} video (can bang fall/non-fall)")

    X, y, subject_list = extract_all_r3d_embeddings(df)

    if len(X) == 0:
        print("[LOI] khong trich xuat duoc embedding nao")
        return

    np.save(output_dir / "X_r3d_features.npy", X)
    np.save(output_dir / "y_r3d_labels.npy", y)

    # ---- Danh gia: train/test split ----
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    clf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr_scaled, y_tr)

    print("\n=== Danh gia tren tap test (25%) ===")
    print(classification_report(y_te, clf.predict(X_te_scaled), target_names=["non-fall", "fall"]))

    # ---- Danh gia: GroupKFold theo subject ----
    print("\n=== Danh gia GroupKFold theo subject (5 fold) ===")
    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(clf, StandardScaler().fit_transform(X), y, groups=subject_list, cv=gkf, scoring="f1")
    print(f"F1 scores: {cv_scores}")
    print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # ---- Luu model (train tren tap da co, giong logic goc trong notebook) ----
    joblib.dump(clf, output_dir / "fall_r3d_classifier.pkl")
    joblib.dump(scaler, output_dir / "r3d_feature_scaler.pkl")
    print(f"\nDa luu model tai: {output_dir}/fall_r3d_classifier.pkl")
    print(f"Da luu scaler tai: {output_dir}/r3d_feature_scaler.pkl")


if __name__ == "__main__":
    main()