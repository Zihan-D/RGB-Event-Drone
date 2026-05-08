import argparse
import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.ppm")


def collect_frames(input_dir: str) -> List[str]:
    files: List[str] = []
    for pattern in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, pattern)))
    files.sort()
    return files


def make_output_size(frame, output_size: Optional[Tuple[int, int]]):
    if output_size is not None:
        return output_size
    h, w = frame.shape[:2]
    return (w, h)


def detect_drone_rgb(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 80, 60], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 18:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)


def detect_event_center(event_frame: np.ndarray) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY) if event_frame.ndim == 3 else event_frame
    _, bin_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)


def draw_overlay(
    frame: np.ndarray,
    index: int,
    total: int,
    mode: str,
    rgb_center: Optional[Tuple[int, int]],
    event_center: Optional[Tuple[int, int]],
):
    cv2.putText(frame, f"Frame {index + 1}/{total}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {mode}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 30), 2)

    if rgb_center is not None:
        cv2.circle(frame, rgb_center, 11, (0, 255, 255), 2)
        cv2.putText(frame, "RGB track", (rgb_center[0] + 12, rgb_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if event_center is not None:
        cv2.circle(frame, event_center, 14, (255, 180, 0), 2)
        cv2.putText(frame, "Event track", (event_center[0] + 12, event_center[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

    if mode == "rgb_event" and rgb_center is not None and event_center is not None:
        fused = ((rgb_center[0] + event_center[0]) // 2, (rgb_center[1] + event_center[1]) // 2)
        cv2.circle(frame, fused, 8, (0, 0, 255), -1)
        cv2.putText(frame, "Fused", (fused[0] + 10, fused[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def reconstruct_video(
    input_dir: str,
    output_path: str,
    fps: int,
    codec: str,
    resize_width: Optional[int],
    resize_height: Optional[int],
    overlay: bool,
    mode: str,
    event_dir: Optional[str],
    rgb_event_background: str,
):
    rgb_frames = collect_frames(input_dir)
    if not rgb_frames:
        raise FileNotFoundError(f"No RGB frames found in: {input_dir}")

    event_frames: List[str] = []
    if mode == "rgb_event":
        if not event_dir:
            raise ValueError("mode=rgb_event requires --event_dir")
        event_frames = collect_frames(event_dir)
        if not event_frames:
            raise FileNotFoundError(f"No event frames found in: {event_dir}")
        if len(event_frames) != len(rgb_frames):
            raise ValueError("RGB and Event frame counts must match for rgb_event mode.")

    first = cv2.imread(rgb_frames[0])
    if first is None:
        raise ValueError(f"Cannot read first RGB frame: {rgb_frames[0]}")

    target_size = (resize_width, resize_height) if resize_width and resize_height else None
    out_w, out_h = make_output_size(first, target_size)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {output_path}; try --codec XVID and .avi")

    total = len(rgb_frames)
    written = 0

    for i, rgb_path in enumerate(rgb_frames):
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"[WARN] skip unreadable RGB frame: {rgb_path}")
            continue

        if (rgb.shape[1], rgb.shape[0]) != (out_w, out_h):
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

        rgb_center = detect_drone_rgb(rgb)
        event_center = None

        if mode == "rgb_event":
            ev = cv2.imread(event_frames[i], cv2.IMREAD_GRAYSCALE)
            if ev is not None:
                if (ev.shape[1], ev.shape[0]) != (out_w, out_h):
                    ev = cv2.resize(ev, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                event_center = detect_event_center(ev)
                if rgb_event_background == "event_gray":
                    rgb = cv2.cvtColor(ev, cv2.COLOR_GRAY2BGR)
                else:
                    # Keep RGB background and slightly blend event intensity into red channel.
                    ev_norm = cv2.normalize(ev, None, 0, 255, cv2.NORM_MINMAX)
                    rgb[:, :, 2] = cv2.addWeighted(rgb[:, :, 2], 0.75, ev_norm, 0.25, 0)

        if overlay:
            draw_overlay(rgb, i, total, mode, rgb_center, event_center)

        writer.write(rgb)
        written += 1

    writer.release()

    if written == 0:
        raise RuntimeError("No frames were written.")

    print("[OK] Video reconstruction completed")
    print(f"Mode: {mode}")
    print(f"RGB input: {input_dir}")
    if mode == "rgb_event":
        print(f"Event input: {event_dir}")
    print(f"Output: {output_path}")
    print(f"Frames written: {written}")
    print(f"FPS: {fps}")
    print(f"Resolution: {out_w}x{out_h}")


def generate_real_background_frame(w: int, h: int, t: int) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y2d = np.repeat(y, w, axis=1)
    x2d = np.repeat(x, h, axis=0)

    sky = (120 + 60 * (1 - y2d)).astype(np.float32)
    ground = (50 + 130 * y2d).astype(np.float32)

    b = np.where(y2d < 0.55, sky + 10 * np.sin(8 * x2d + 0.08 * t), ground)
    g = np.where(y2d < 0.55, sky * 0.95, ground + 15 * np.sin(6 * x2d + 0.05 * t))
    r = np.where(y2d < 0.55, sky * 0.7, ground * 0.8)

    img = np.stack([b, g, r], axis=2)
    noise = np.random.normal(0, 6, size=(h, w, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # roads/buildings style shapes
    cv2.rectangle(img, (0, int(h * 0.72)), (w, h), (70, 95, 80), -1)
    cv2.rectangle(img, (int(w * 0.15), int(h * 0.58)), (int(w * 0.22), int(h * 0.72)), (90, 100, 110), -1)
    cv2.rectangle(img, (int(w * 0.63), int(h * 0.6)), (int(w * 0.75), int(h * 0.72)), (88, 96, 108), -1)
    cv2.line(img, (0, int(h * 0.85)), (w, int(h * 0.8)), (120, 130, 120), 2)

    return img


def generate_demo_dataset(base_dir: str, n: int = 90, w: int = 960, h: int = 540):
    rgb_real_dir = os.path.join(base_dir, "rgb_real")
    rgb_event_rgb_dir = os.path.join(base_dir, "rgb_event", "rgb")
    rgb_event_event_dir = os.path.join(base_dir, "rgb_event", "event")

    os.makedirs(rgb_real_dir, exist_ok=True)
    os.makedirs(rgb_event_rgb_dir, exist_ok=True)
    os.makedirs(rgb_event_event_dir, exist_ok=True)

    prev_gray = None
    for i in range(n):
        base = generate_real_background_frame(w, h, i)

        x = int(80 + (w - 160) * i / (n - 1))
        y = int(h * 0.38 + 120 * np.sin(i * 0.13))

        # drone marker: small green rotor-cross
        cv2.circle(base, (x, y), 7, (30, 230, 50), -1)
        cv2.line(base, (x - 10, y), (x + 10, y), (40, 250, 70), 2)
        cv2.line(base, (x, y - 10), (x, y + 10), (40, 250, 70), 2)

        # target marker in realistic scene
        tx = int(w * 0.7 + 95 * np.sin(i * 0.11))
        ty = int(h * 0.55 + 70 * np.cos(i * 0.15))
        cv2.rectangle(base, (tx - 7, ty - 7), (tx + 7, ty + 7), (40, 40, 220), -1)

        rgb_real_path = os.path.join(rgb_real_dir, f"frame_{i:04d}.png")
        cv2.imwrite(rgb_real_path, base)

        rgb_event_rgb_path = os.path.join(rgb_event_rgb_dir, f"frame_{i:04d}.png")
        cv2.imwrite(rgb_event_rgb_path, base)

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            event = np.zeros_like(gray)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            _, event = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            # reinforce drone movement in event map
            cv2.circle(event, (x, y), 10, 255, 1)

        event_colored = cv2.cvtColor(event, cv2.COLOR_GRAY2BGR)
        rgb_event_event_path = os.path.join(rgb_event_event_dir, f"frame_{i:04d}.png")
        cv2.imwrite(rgb_event_event_path, event_colored)
        prev_gray = gray

    print("[OK] Demo dataset generated")
    print(f"RGB real dir: {rgb_real_dir}")
    print(f"RGB-Event RGB dir: {rgb_event_rgb_dir}")
    print(f"RGB-Event Event dir: {rgb_event_event_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Video reconstruction and tracking demo for drone projects.")
    parser.add_argument("--input_dir", help="RGB frame directory")
    parser.add_argument("--event_dir", default=None, help="Event frame directory (for mode=rgb_event)")
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--resize_width", type=int, default=None)
    parser.add_argument("--resize_height", type=int, default=None)
    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--mode", choices=["rgb_real", "rgb_event"], default="rgb_real")
    parser.add_argument(
        "--rgb_event_background",
        choices=["event_gray", "rgb"],
        default="event_gray",
        help="Background style when mode=rgb_event (default: event_gray)",
    )
    parser.add_argument("--prepare_demo", action="store_true", help="Generate demo frames for testing")
    parser.add_argument("--demo_dir", default="test_assets/tracking_demo", help="Demo dataset root dir")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prepare_demo:
        generate_demo_dataset(args.demo_dir)
        return

    if not args.input_dir or not args.output:
        raise SystemExit("--input_dir and --output are required unless using --prepare_demo")

    reconstruct_video(
        input_dir=args.input_dir,
        output_path=args.output,
        fps=args.fps,
        codec=args.codec,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        overlay=not args.no_overlay,
        mode=args.mode,
        event_dir=args.event_dir,
        rgb_event_background=args.rgb_event_background,
    )


if __name__ == "__main__":
    main()
