#!/usr/bin/env python3
import os
import sys
import time
import random
import argparse
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, as_completed

script_dir = Path(__file__).resolve().parent

def render_single_skin(args):
    skin_path, output_path, renderer_path, ddj_path, bg_color, force = args
    if not force and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True, skin_path, None
    try:
        if renderer_path not in sys.path:
            sys.path.insert(0, renderer_path)
        if ddj_path not in sys.path:
            sys.path.insert(0, ddj_path)
        from create_template import create_template

        devnull = open(os.devnull, 'w')
        with redirect_stdout(devnull), redirect_stderr(devnull):
            create_template(
                skin_path=skin_path,
                output_path=output_path,
                renderer_path=renderer_path,
                size="512x1024",
                bg_color=bg_color
            )
        devnull.close()
        return True, skin_path, None
    except Exception as e:
        return False, skin_path, str(e)

def main():
    parser = argparse.ArgumentParser(description="Render skins to control_imgs with random background colors")
    parser.add_argument("--num_skins", type=int, default=1000, help="Number of skins to render")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", default=False, help="Force overwrite existing images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for background colors")
    parser.add_argument("--skins_dir", type=str, required=True, help="Path to skins directory")
    parser.add_argument("--ddj_path", type=str, required=True, help="Path to DDJ_real2render directory containing create_template.py")
    parser.add_argument("--renderer_path", "--differentiable_minecraft_renderer_path", type=str, required=True, help="Path to differentiable_minecraft_renderer directory")
    args = parser.parse_args()

    ddj_path = os.path.abspath(args.ddj_path)
    if ddj_path not in sys.path:
        sys.path.insert(0, ddj_path)

    renderer_path = os.path.abspath(args.renderer_path)
    if renderer_path not in sys.path:
        sys.path.insert(0, renderer_path)

    skins_dir = Path(args.skins_dir).resolve()
    control_imgs_dir = script_dir / "control_imgs"
    control_imgs_dir.mkdir(parents=True, exist_ok=True)

    # List and sort skins numerically
    skin_files = [f for f in os.listdir(skins_dir) if f.endswith(".png") and f[:-4].isdigit()]
    skin_files.sort(key=lambda x: int(x[:-4]))
    selected_skins = skin_files[:args.num_skins]

    print(f"Found {len(skin_files)} total skins. Selected first {len(selected_skins)} skins for rendering.")
    print(f"Output directory: {control_imgs_dir}")
    print(f"DDJ path: {ddj_path}")
    print(f"Renderer path: {renderer_path}")
    print(f"Background colors: Random RGB per skin (seed={args.seed})")

    tasks = []
    for skin_file in selected_skins:
        skin_id = int(skin_file[:-4])
        # Deterministic random color per skin ID
        rng = random.Random(args.seed + skin_id)
        bg_color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        
        skin_path = str(skins_dir / skin_file)
        output_path = str(control_imgs_dir / skin_file)
        tasks.append((skin_path, output_path, renderer_path, ddj_path, bg_color, args.force))

    start_time = time.time()
    success_count = 0
    fail_count = 0

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(render_single_skin, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                success, path, err = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"Failed {path}: {err}")
                if i % 25 == 0 or i == len(tasks):
                    elapsed = time.time() - start_time
                    fps = i / elapsed
                    eta = (len(tasks) - i) / fps if fps > 0 else 0
                    print(f"Progress: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) - {fps:.2f} img/s - ETA: {eta:.1f}s", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            success, path, err = render_single_skin(task)
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Failed {path}: {err}")
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                fps = i / elapsed
                eta = (len(tasks) - i) / fps if fps > 0 else 0
                print(f"Progress: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) - {fps:.2f} img/s - ETA: {eta:.1f}s", flush=True)

    total_time = time.time() - start_time
    print(f"\nDone! Successfully rendered {success_count}/{len(tasks)} skins with random background colors in {total_time:.2f}s ({fail_count} failed).", flush=True)

if __name__ == "__main__":
    main()
