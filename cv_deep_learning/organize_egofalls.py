"""
organize_egofalls.py — Sap xep dataset EGOFALLS da giai nen vao data/normal/ va data/fall/

CAU TRUC GOC DA XAC NHAN TU SERVER THAT (khong phai doan):
    EGOFALLS/{subject}/extracted/{S_X}/{Indoor|Outdoor}/{falls|nonfalls}/{HoatDong}/.../{Neck|Waist}/*.mp4

LUU Y QUAN TRONG DA XU LY: chu "nonfalls" CHUA chuoi con "falls" ben trong
no (non-FALLS) - neu chi kiem tra "falls" in path se phan loai NHAM toan bo
video nonfalls thanh fall. Script nay so khop CHINH XAC tung doan thu muc
(path segment), khong so khop chuoi con.

MAC DINH DUNG SYMLINK (khong copy file that) de tiet kiem dung luong dia -
dataset EGOFALLS ~200GB, copy het se ton gap doi dung luong khong can thiet.

CACH CHAY:
    python organize_egofalls.py --source EGOFALLS --dest data --mode symlink

    # Neu chi muon lay video Outdoor (gan voi bai toan xe may hon Indoor):
    python organize_egofalls.py --source EGOFALLS --dest data --env outdoor

Ket qua: data/normal/*.mp4 va data/fall/*.mp4 (thuc chat la symlink tro ve
file goc trong EGOFALLS/), kem manifest.csv ghi lai day du metadata
(subject, environment, camera_position, nhan) cho tung video.
"""

import argparse
import csv
import os
from pathlib import Path


def classify_label(path_parts: list[str]) -> str | None:
    """So khop CHINH XAC tung doan duong dan, tranh loi 'nonfalls' chua chuoi con 'falls'."""
    lower_parts = [p.lower() for p in path_parts]
    if "nonfalls" in lower_parts:
        return "normal"
    if "falls" in lower_parts:
        return "fall"
    return None  # khong xac dinh duoc nhan tu duong dan - se bao cao rieng, khong bo qua am tham


def get_environment(path_parts: list[str]) -> str:
    lower_parts = [p.lower() for p in path_parts]
    if "outdoor" in lower_parts:
        return "outdoor"
    if "indoor" in lower_parts:
        return "indoor"
    return "unknown"


def get_camera_position(path_parts: list[str]) -> str:
    lower_parts = [p.lower() for p in path_parts]
    if "neck" in lower_parts:
        return "neck"
    if "waist" in lower_parts:
        return "waist"
    return "unknown"


def get_subject(path_parts: list[str]) -> str:
    """Subject thuong la doan dang S_X ngay sau 'extracted' - lay doan dau tien khop pattern S_<chu>."""
    for p in path_parts:
        if p.startswith("S_") and len(p) <= 6:  # vd S_L, S_R, S_Q, S_XL...
            return p
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Sap xep dataset EGOFALLS vao data/normal va data/fall")
    parser.add_argument("--source", default="EGOFALLS", help="Thu muc goc chua EGOFALLS da giai nen")
    parser.add_argument("--dest", default="data", help="Thu muc dich (se tao dest/normal va dest/fall)")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                         help="symlink (mac dinh, tiet kiem dia) hoac copy (ban that, ton nhieu dia hon)")
    parser.add_argument("--env", choices=["all", "indoor", "outdoor"], default="all",
                         help="Loc theo moi truong - 'outdoor' gan voi bai toan xe may hon 'indoor'")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest_normal = Path(args.dest) / "normal"
    dest_fall = Path(args.dest) / "fall"
    dest_normal.mkdir(parents=True, exist_ok=True)
    dest_fall.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.dest) / "egofalls_manifest.csv"
    manifest_rows = []

    unclassified = []
    count_normal, count_fall, count_skipped_env = 0, 0, 0

    video_files = list(source.rglob("*.mp4"))
    print(f"Tim thay {len(video_files)} file .mp4 trong {source}")

    for video_path in video_files:
        rel_parts = video_path.relative_to(source).parts

        label = classify_label(rel_parts)
        if label is None:
            unclassified.append(str(video_path))
            continue

        environment = get_environment(rel_parts)
        if args.env != "all" and environment != args.env:
            count_skipped_env += 1
            continue

        camera_position = get_camera_position(rel_parts)
        subject = get_subject(rel_parts)

        # dat ten file dich duy nhat, tranh trung ten giua cac subject/hoat dong khac nhau
        dest_name = f"{subject}_{environment}_{camera_position}_{video_path.stem}.mp4"
        dest_dir = dest_normal if label == "normal" else dest_fall
        dest_path = dest_dir / dest_name

        if args.mode == "symlink":
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            os.symlink(video_path, dest_path)
        else:
            import shutil
            shutil.copy2(video_path, dest_path)

        if label == "normal":
            count_normal += 1
        else:
            count_fall += 1

        manifest_rows.append({
            "dest_filename": dest_name,
            "label": label,
            "environment": environment,
            "camera_position": camera_position,
            "subject": subject,
            "source_path": str(video_path),
        })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dest_filename", "label", "environment", "camera_position", "subject", "source_path"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n--- Ket qua ---")
    print(f"  normal: {count_normal} video -> {dest_normal}")
    print(f"  fall:   {count_fall} video -> {dest_fall}")
    if count_skipped_env:
        print(f"  bo qua do loc moi truong ('{args.env}'): {count_skipped_env} video")
    if unclassified:
        print(f"\n[CANH BAO] {len(unclassified)} video KHONG xac dinh duoc nhan (khong co 'falls'/'nonfalls' trong duong dan):")
        for u in unclassified[:10]:
            print(f"    {u}")
        if len(unclassified) > 10:
            print(f"    ... va {len(unclassified) - 10} file khac")
    print(f"\nManifest day du (metadata tung video) da luu tai: {manifest_path}")


if __name__ == "__main__":
    main()