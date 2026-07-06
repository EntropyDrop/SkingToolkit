#!/usr/bin/env python3
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from build_target_img import build_target_img

def process_single_skin(args):
    skin_path, output_path, force = args
    if force and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass
    elif not force and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True, skin_path, None
    try:
        build_target_img(skin_path, output_path)
        return True, skin_path, None
    except Exception as e:
        return False, skin_path, str(e)

def main():
    parser = argparse.ArgumentParser(description="Batch process Minecraft skins to target images using build_target_img.py")
    parser.add_argument("--skins_dir", "--skins", type=str, required=True, help="Path to input skins directory")
    parser.add_argument("--num_skins", type=int, default=1000, help="Number of skins to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for target images")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", default=False, help="Force overwrite existing target images")
    args = parser.parse_args()

    skins_dir = Path(args.skins_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skins_dir.exists():
        print(f"Error: skins_dir '{skins_dir}' does not exist.")
        sys.exit(1)

    # List and sort skin files numerically
    skin_files = [f for f in os.listdir(skins_dir) if f.endswith(".png") and f[:-4].isdigit()]
    skin_files.sort(key=lambda x: int(x[:-4]))
    selected_skins = skin_files[:args.num_skins]

    print(f"Found {len(skin_files)} total skins. Selected first {len(selected_skins)} skins.")
    print(f"Skins input directory: {skins_dir}")
    print(f"Output target images directory: {output_dir}")

    tasks = []
    for skin_file in selected_skins:
        skin_path = str(skins_dir / skin_file)
        output_path = str(output_dir / skin_file)
        tasks.append((skin_path, output_path, args.force))

    start_time = time.time()
    success_count = 0
    fail_count = 0

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_skin, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                success, path, err = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"Failed {path}: {err}")
                if i % 100 == 0 or i == len(tasks):
                    elapsed = time.time() - start_time
                    fps = i / elapsed
                    eta = (len(tasks) - i) / fps if fps > 0 else 0
                    print(f"Progress: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) - {fps:.2f} img/s - ETA: {eta:.1f}s", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            success, path, err = process_single_skin(task)
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Failed {path}: {err}")
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                fps = i / elapsed
                eta = (len(tasks) - i) / fps if fps > 0 else 0
                print(f"Progress: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) - {fps:.2f} img/s - ETA: {eta:.1f}s", flush=True)

    total_time = time.time() - start_time
    print(f"\nDone! Successfully processed {success_count}/{len(tasks)} target images in {total_time:.2f}s ({fail_count} failed).", flush=True)

if __name__ == "__main__":
    main()
