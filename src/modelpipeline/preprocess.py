import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config

def sample_indices(total_frames: int, n: int) -> list:
    if total_frames <= n:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, n, dtype=int).tolist()

def process_video(args):
    """
    Extracts and resizes frames for ViT (224x224).
    """
    vid_path, label, data_root, output_root, frames_per_video, face_size = args

    rel_path = Path(vid_path).relative_to(data_root)
    frame_dir = Path(output_root) / rel_path.parent / rel_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(vid_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = sample_indices(total, frames_per_video)
    results = []

    for i, idx in enumerate(indices):
        out_path = frame_dir / f"frame_{i:04d}.jpg"

        if out_path.exists():
            rel_frame = out_path.relative_to(output_root)
            results.append((str(rel_frame), label))
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Convert to RGB for consistency with Transformer training
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ViT Requirement: Resize to 224x224
        # INTER_CUBIC is used for better detail retention in high-frequency areas
        frame = cv2.resize(frame, (face_size, face_size), interpolation=cv2.INTER_CUBIC)

        # Save as high-quality BGR for OpenCV imwrite
        save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), save_frame, [cv2.IMWRITE_JPEG_QUALITY, 98])

        rel_frame = out_path.relative_to(output_root)
        results.append((str(rel_frame), label))

    cap.release()
    return results

def preprocess_dataset(cfg):
    output_root = Path(cfg.FRAMES_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.csv"

    # Ensure metadata path is correct relative to DATA_ROOT
    meta_path = Path(cfg.DATA_ROOT) / "csv" / "FF++_Metadata.csv"
    if not meta_path.exists():
        print(f"[!] Metadata not found at {meta_path}")
        return

    df = pd.read_csv(meta_path, index_col=0)[["File Path", "Label"]].dropna()

    allowed_prefixes = tuple(["original"] + cfg.MANIPULATION_TYPES)
    df = df[df["File Path"].str.startswith(allowed_prefixes)].reset_index(drop=True)

    tasks = []
    for _, row in df.iterrows():
        vid_path = Path(cfg.DATA_ROOT) / row["File Path"]
        if vid_path.exists():
            tasks.append((
                str(vid_path),
                row["Label"],
                str(cfg.DATA_ROOT),
                str(output_root),
                cfg.FRAMES_PER_VIDEO,
                cfg.FACE_SIZE,
            ))

    print(f"[*] Videos found: {len(tasks)}")
    print(f"[*] Target Size: {cfg.FACE_SIZE}x{cfg.FACE_SIZE}")

    all_records = []
    failed = 0

    with ThreadPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
        futures = {executor.submit(process_video, t): t for t in tasks}
        bar = tqdm(as_completed(futures), total=len(futures),
                   desc="Processing for ViT", unit="video")
        for future in bar:
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                failed += 1
                bar.set_postfix(failed=failed)

    manifest = pd.DataFrame(all_records, columns=["Frame Path", "Label"])
    manifest.to_csv(manifest_path, index=False)

    print(f"\n[*] Complete. {len(manifest)} frames saved to {manifest_path}")
    if failed:
        print(f"[!] Failed videos: {failed}")

if __name__ == "__main__":
    cfg = Config()
    preprocess_dataset(cfg)
