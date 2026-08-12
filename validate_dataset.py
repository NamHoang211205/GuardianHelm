"""
validate_dataset.py — Quet toan bo data/normal/ va data/fall/ de tim file
video bi hong (corrupted), tach rieng ra truoc khi train, tranh loi
"sps_id out of range"/"Invalid NAL unit size" lam nhieu log va co the anh
huong ket qua train.

CACH CHAY:
    python3 validate_dataset.py --data_dir ./data --quarantine_dir ./data_corrupted

Sau khi chay xong: file hong se duoc CHUYEN (move, khong phai xoa) sang
--quarantine_dir, giu nguyen cau truc normal/fall - co the xem lai sau,
khong mat du lieu vinh vien.
"""

import argparse
import shutil
from pathlib import Path

import cv2


def is_video_corrupted(path: Path, min_readable_frames: int = 3) -> tuple[bool, str]:
    """
    Kiem tra 1 video co doc duoc khong. Tra ve (co_hong, ly_do).
    Chi can doc thu vai frame dau, khong can doc het toan bo video - du de
    phat hien phan lon truong hop hong ma khong ton qua nhieu thoi gian.
    """
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return True, "khong mo duoc file"

    total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_reported <= 0:
        cap.release()
        return True, "metadata bao 0 frame (file co the bi hong phan header)"

    readable_count = 0
    for _ in range(min(min_readable_frames, total_frames_reported)):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        readable_count += 1

    cap.release()

    if readable_count < min_readable_frames and readable_count < total_frames_reported:
        return True, f"chi doc duoc {readable_count}/{min_readable_frames} frame thu nghiem"

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Quet va cach ly video hong truoc khi train")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--quarantine_dir", default="./data_corrupted")
    parser.add_argument("--dry_run", action="store_true",
                         help="Chi bao cao, KHONG di chuyen file (dung de xem truoc)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    quarantine_dir = Path(args.quarantine_dir)

    corrupted_report = []

    for label in ["normal", "fall"]:
        src_dir = data_dir / label
        if not src_dir.exists():
            print(f"[CANH BAO] khong tim thay {src_dir}, bo qua")
            continue

        videos = list(src_dir.glob("*.mp4"))
        print(f"\nDang kiem tra {len(videos)} video trong {src_dir}...")

        for i, v in enumerate(videos, 1):
            if i % 200 == 0:
                print(f"  ... da kiem tra {i}/{len(videos)}")

            corrupted, reason = is_video_corrupted(v)
            if corrupted:
                corrupted_report.append((str(v), label, reason))
                if not args.dry_run:
                    dest_dir = quarantine_dir / label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    # neu la symlink (tao boi organize_egofalls.py), chi xoa symlink,
                    # KHONG dong cham gi den file goc trong EGOFALLS/
                    if v.is_symlink():
                        v.unlink()
                    else:
                        shutil.move(str(v), str(dest_dir / v.name))

    print(f"\n=== KET QUA ===")
    print(f"Tong so video hong phat hien: {len(corrupted_report)}")
    if corrupted_report:
        print("\nChi tiet:")
        for path, label, reason in corrupted_report[:30]:
            print(f"  [{label}] {Path(path).name}: {reason}")
        if len(corrupted_report) > 30:
            print(f"  ... va {len(corrupted_report) - 30} file khac")

        if not args.dry_run:
            print(f"\nDa chuyen {len(corrupted_report)} file hong sang: {quarantine_dir}")
        else:
            print("\n(--dry_run: chua di chuyen file nao, chi bao cao)")
    else:
        print("Khong phat hien file hong nao - dataset sach, san sang train.")


if __name__ == "__main__":
    main()