"""Display MNIST test-set images based on YAML settings using OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

try:
    from torchvision import datasets
except ImportError as exc:
    raise ImportError(
        "PyTorch and torchvision are required to load MNIST. Install with: pip install torch torchvision"
    ) from exc


def parse_digits(digit_config: object) -> set[int] | None:
    """Parse digit setting.

    Returns:
        None if all digits should be displayed, otherwise a set of digits.
    """
    if isinstance(digit_config, str):
        if digit_config.lower() == "all":
            return None
        if digit_config.isdigit():
            digit_value = int(digit_config)
            if 0 <= digit_value <= 9:
                return {digit_value}
        raise ValueError("'digit' string must be 'all' or a digit from 0 to 9")

    if isinstance(digit_config, int):
        if 0 <= digit_config <= 9:
            return {digit_config}
        raise ValueError("'digit' integer must be in range 0..9")

    if isinstance(digit_config, (list, tuple)):
        if not digit_config:
            raise ValueError("'digit' list cannot be empty")
        parsed_digits: set[int] = set()
        for item in digit_config:
            if isinstance(item, str) and item.isdigit():
                item = int(item)
            if not isinstance(item, int) or not (0 <= item <= 9):
                raise ValueError("'digit' list items must be integers in range 0..9")
            parsed_digits.add(item)
        return parsed_digits

    raise ValueError("Unsupported 'digit' value. Use an int, list, or 'all'.")


def load_settings(
    settings_path: Path,
) -> tuple[
    float,
    set[int] | None,
    int,
    int,
    int,
    float,
    int,
    str,
    bool,
    tuple[int, int] | None,
    int,
    int,
]:
    with settings_path.open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}

    display_time_ms = settings.get("display_time", 700)
    digit_config = settings.get("digit", "all")
    width_px = settings.get("w", 640)
    height_px = settings.get("h", 640)
    padding_px = settings.get("padding", 0)
    frame_time_ms = settings.get("frame_time", 50)
    movement_distance_px = settings.get("movement_distance", 20)
    movement_mode = settings.get("movement_mode", "diamond")
    shuffle_samples = settings.get("shuffle", False)
    sample_range_config = settings.get("sample_range", "all")

    window_x = settings.get("win_x", 100)
    window_y = settings.get("win_y", 100)

    try:
        display_time_ms = float(display_time_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("'display_time' must be a numeric value in milliseconds") from exc

    if display_time_ms <= 0:
        raise ValueError("'display_time' must be > 0")

    try:
        frame_time_ms = float(frame_time_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("'frame_time' must be a numeric value in milliseconds") from exc

    if frame_time_ms <= 0:
        raise ValueError("'frame_time' must be > 0")

    if not isinstance(movement_mode, str):
        raise ValueError(
            "'movement_mode' must be a string: 'none', 'diamond', 'hourglass', or 'diagonal'"
        )
    movement_mode = movement_mode.lower()
    if movement_mode not in {"none", "diamond", "diamond-ccw", "diamond-ccw-step1", "hourglass", "diagonal", "legacy"}:
        raise ValueError(
            "'movement_mode' must be one of: 'none', 'diamond', 'diamond-ccw', 'diamond-ccw-step1', 'hourglass', 'diagonal', 'legacy'"
        )

    try:
        width_px = int(width_px)
        height_px = int(height_px)
        padding_px = int(padding_px)
        movement_distance_px = int(movement_distance_px)
        window_x = int(window_x)
        window_y = int(window_y)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "'w', 'h', 'padding', 'movement_distance', 'window_x', and 'window_y' must be integers"
        ) from exc

    if width_px <= 0 or height_px <= 0:
        raise ValueError("'w' and 'h' must be > 0")

    if padding_px < 0:
        raise ValueError("'padding' must be >= 0")
    if padding_px * 2 >= min(width_px, height_px):
        raise ValueError("'padding' is too large for the selected 'w' and 'h'")
    if movement_distance_px < 0:
        raise ValueError("'movement_distance' must be >= 0")

    if not isinstance(shuffle_samples, bool):
        raise ValueError("'shuffle' must be true or false")

    sample_range: tuple[int, int] | None
    if sample_range_config is None or sample_range_config == "all":
        sample_range = None
    elif isinstance(sample_range_config, (list, tuple)) and len(sample_range_config) == 2:
        start_idx, end_idx = sample_range_config
        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            raise ValueError("'sample_range' must be [start, end] with integer indices")
        if start_idx < 0 or end_idx < 0:
            raise ValueError("'sample_range' indices must be >= 0")
        if start_idx > end_idx:
            raise ValueError("'sample_range' requires start <= end")
        sample_range = (start_idx, end_idx)
    else:
        raise ValueError("'sample_range' must be 'all' or [start, end]")

    display_time_seconds = display_time_ms / 1000.0
    frame_time_seconds = frame_time_ms / 1000.0
    selected_digits = parse_digits(digit_config)

    return (
        display_time_seconds,
        selected_digits,
        width_px,
        height_px,
        padding_px,
        frame_time_seconds,
        movement_distance_px,
        movement_mode,
        shuffle_samples,
        sample_range,
        window_x,
        window_y,
    )


def select_samples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected_digits: set[int] | None,
    shuffle_samples: bool,
    sample_range: tuple[int, int] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    if selected_digits is None:
        mask = np.ones_like(y_test, dtype=bool)
        selected_desc = "all digits"
    else:
        mask = np.isin(y_test, list(selected_digits))
        selected_desc = f"digits {sorted(selected_digits)}"

    selected_images = x_test[mask]
    selected_labels = y_test[mask]

    if shuffle_samples and selected_images.size > 0:
        permutation = rng.permutation(len(selected_images))
        selected_images = selected_images[permutation]
        selected_labels = selected_labels[permutation]

    range_desc = "all"
    if sample_range is not None:
        start_idx, end_idx = sample_range
        selected_images = selected_images[start_idx : end_idx + 1]
        selected_labels = selected_labels[start_idx : end_idx + 1]
        range_desc = f"{start_idx}..{end_idx}"

    selected_desc = f"{selected_desc}, shuffle={shuffle_samples}, range={range_desc}"
    return selected_images, selected_labels, selected_desc


def build_layout(
    width_px: int,
    height_px: int,
    padding_px: int,
    movement_distance_px: int,
    movement_mode: str,
    min_window_size: int,
    title_band_px: int,
) -> tuple[int, int, int, int, int, int, list[tuple[int, int]]]:
    window_width_px = max(width_px, min_window_size)
    window_height_px = max(height_px + title_band_px, min_window_size)

    available_height_px = window_height_px - title_band_px
    content_left_px = int((window_width_px - width_px) / 2.0)
    content_top_px = int((available_height_px - height_px) / 2.0)

    image_left_px = content_left_px + padding_px
    image_top_px = content_top_px + padding_px
    image_width_px = width_px - (2 * padding_px)
    image_height_px = height_px - (2 * padding_px)

    max_movement_x = max(0, content_left_px)
    max_movement_y = max(0, content_top_px)
    effective_movement_distance_px = min(
        movement_distance_px,
        max_movement_x,
        max_movement_y,
    )

    if movement_mode == "none":
        movement_positions = [(0, 0)]
    elif movement_mode == "diamond":
        movement_positions = [
            (0, -effective_movement_distance_px),
            (effective_movement_distance_px, 0),
            (0, effective_movement_distance_px),
            (-effective_movement_distance_px, 0),
        ]
    elif movement_mode == "diamond-ccw":
        movement_positions = [
            (effective_movement_distance_px, 0),
            (0, -effective_movement_distance_px),
            (-effective_movement_distance_px, 0),
            (0, effective_movement_distance_px),
        ]
    elif movement_mode == "diamond-ccw-step1":
        d = effective_movement_distance_px
        movement_positions = []

        # Start at (d, 0)

        # 1. Right → Top (move up-left)
        for i in range(d):
            movement_positions.append((d - i, -i))

        # 2. Top → Left (move down-left)
        for i in range(d):
            movement_positions.append((0 - i, -d + i))

        # 3. Left → Bottom (move down-right)
        for i in range(d):
            movement_positions.append((-d + i, 0 + i))

        # 4. Bottom → Right (move up-right)
        for i in range(d):
            movement_positions.append((0 + i, d - i))
    elif movement_mode == "hourglass":
        movement_positions = [
            (-effective_movement_distance_px, -effective_movement_distance_px),
            (effective_movement_distance_px, effective_movement_distance_px),
            (effective_movement_distance_px, -effective_movement_distance_px),
            (-effective_movement_distance_px, effective_movement_distance_px),
        ]
    elif movement_mode == "legacy":
        y_range = movement_distance_px
        x1_range = movement_distance_px
        x2_range = movement_distance_px

        movement_positions = []

        # Phase 1: vertical down
        for y in range(0, y_range, 2):
            movement_positions.append((0, y))

        # Phase 2: diagonal up-right
        for x, y in zip(range(0, y_range, 3), range(0, x1_range, 2)):
            movement_positions.append((x, y_range - y))

        # Phase 3: horizontal left
        for x in range(0, x2_range, 2):
            movement_positions.append((x1_range - x, 0))
    else:
        movement_positions = [
            (-effective_movement_distance_px, -effective_movement_distance_px),
            (effective_movement_distance_px, effective_movement_distance_px),
        ]

    return (
        window_width_px,
        window_height_px,
        image_left_px,
        image_top_px,
        image_width_px,
        image_height_px,
        movement_positions,
    )


def make_frame(
    image: np.ndarray | None,
    label: int | None,
    sample_idx: int,
    total_samples: int,
    movement_mode: str,
    window_width_px: int,
    window_height_px: int,
    image_left_px: int,
    image_top_px: int,
    image_width_px: int,
    image_height_px: int,
    offset_x: int,
    offset_y: int,
    title_band_px: int,
    message: str | None = None,
) -> np.ndarray:
    frame = np.zeros((window_height_px, window_width_px, 3), dtype=np.uint8)
    if movement_mode == "legacy":
        frame = np.zeros((window_height_px, window_width_px), dtype=np.uint8)
    else:
        frame = np.zeros((window_height_px, window_width_px, 3), dtype=np.uint8)
    if image is not None:
        resized = cv2.resize(
            image,
            (image_width_px, image_height_px),
            # interpolation=cv2.INTER_NEAREST,
        )
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        x0 = image_left_px + offset_x
        y0 = image_top_px + offset_y
        x1 = x0 + image_width_px
        y1 = y0 + image_height_px

        if 0 <= x0 < window_width_px and 0 <= y0 < window_height_px and x1 <= window_width_px and y1 <= window_height_px:
            if movement_mode == "legacy":
                frame[y0:y1, x0:x1] = resized  # ✅ grayscale → grayscale
            else:
                resized_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                frame[y0:y1, x0:x1] = resized_bgr  # ✅ BGR → BGR

    if message is not None:
        title = message
    else:
        title = f"label: {label}" #f"MNIST test sample {sample_idx + 1}/{total_samples} - label: {label}"

    text_y = window_height_px - title_band_px + 35
    cv2.putText(
        frame,
        title,
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def main() -> None:
    output_path = Path("experiments") / "digit_times.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    parser = argparse.ArgumentParser(description="Display MNIST test set samples with OpenCV")
    parser.add_argument(
        "--settings",
        default="mnist_settings.yaml",
        help="Path to YAML settings file (default: mnist_settings.yaml)",
    )
    args = parser.parse_args()

    settings_path = Path(args.settings)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    (
        display_time_seconds,
        selected_digits,
        width_px,
        height_px,
        padding_px,
        frame_time_seconds,
        movement_distance_px,
        movement_mode,
        shuffle_samples,
        sample_range,
        window_x,
        window_y,
    ) = load_settings(settings_path)

    dataset = datasets.MNIST(root="data", train=False, download=True)
    x_test = dataset.data.numpy()
    y_test = dataset.targets.numpy()

    rng = np.random.default_rng()

    selected_images, selected_labels, selected_desc = select_samples(
        x_test,
        y_test,
        selected_digits,
        shuffle_samples,
        sample_range,
        rng,
    )

    if selected_images.size == 0:
        print("No images found for the selected digit filter.")
        return

    print(
        f"Displaying {len(selected_images)} test samples for {selected_desc} "
        f"with display_time={display_time_seconds * 1000:.0f}ms at size {width_px}x{height_px} "
        f"and padding={padding_px}px (frame_time={frame_time_seconds * 1000:.0f}ms, "
        f"movement_distance={movement_distance_px}px, movement_mode={movement_mode}, "
        f"shuffle={shuffle_samples}, sample_range={sample_range if sample_range is not None else 'all'})"
    )

    min_window_size = 500
    title_band_px = 90

    (
        window_width_px,
        window_height_px,
        image_left_px,
        image_top_px,
        image_width_px,
        image_height_px,
        movement_positions,
    ) = build_layout(
        width_px,
        height_px,
        padding_px,
        movement_distance_px,
        movement_mode,
        min_window_size,
        title_band_px,
    )

    window_name = "MNIST Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width_px, window_height_px)
    cv2.moveWindow(window_name, window_x, window_y)

    start_frame = make_frame(
        image=None,
        label=None,
        sample_idx=0,
        total_samples=max(len(selected_images), 1),
        movement_mode=movement_mode,
        window_width_px=window_width_px,
        window_height_px=window_height_px,
        image_left_px=image_left_px,
        image_top_px=image_top_px,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        offset_x=0,
        offset_y=0,
        title_band_px=title_band_px,
        message="" #"Press SPACE to start (ESC or Q to quit)",
    )
    cv2.imshow(window_name, start_frame)

    started = False
    while not started:
        key = cv2.waitKey(50) & 0xFF
        if key == 27 or key == ord("q"):
            cv2.destroyAllWindows()
            print("Display window closed by user. Stopping.")
            return
        if key == 32:
            started = True

    try:
        last_settings_mtime_ns = settings_path.stat().st_mtime_ns
    except OSError:
        last_settings_mtime_ns = -1

    sample_idx = 0

    silence_ms = 3_000  # Silence period in milliseconds

    while sample_idx < len(selected_images):
        try:
            current_mtime_ns = settings_path.stat().st_mtime_ns
        except OSError:
            current_mtime_ns = last_settings_mtime_ns

        if current_mtime_ns != last_settings_mtime_ns:
            try:
                previous_shuffle_samples = shuffle_samples
                previous_sample_range = sample_range

                (
                    display_time_seconds,
                    selected_digits,
                    width_px,
                    height_px,
                    padding_px,
                    frame_time_seconds,
                    movement_distance_px,
                    movement_mode,
                    shuffle_samples,
                    sample_range,
                    window_x,
                    window_y,
                ) = load_settings(settings_path)
            except ValueError as exc:
                print(f"Ignoring invalid settings update: {exc}")
            else:
                restart_sequence = (
                    shuffle_samples != previous_shuffle_samples
                    or sample_range != previous_sample_range
                )

                selected_images, selected_labels, selected_desc = select_samples(
                    x_test,
                    y_test,
                    selected_digits,
                    shuffle_samples,
                    sample_range,
                    rng,
                )

                (
                    window_width_px,
                    window_height_px,
                    image_left_px,
                    image_top_px,
                    image_width_px,
                    image_height_px,
                    movement_positions,
                ) = build_layout(
                    width_px,
                    height_px,
                    padding_px,
                    movement_distance_px,
                    movement_mode,
                    min_window_size,
                    title_band_px,
                )

                cv2.resizeWindow(window_name, window_width_px, window_height_px)
                cv2.moveWindow(window_name, window_x, window_y)

                if selected_images.size == 0:
                    sample_idx = 0
                    print("Settings reloaded: no images for selected digit filter.")
                else:
                    if restart_sequence:
                        sample_idx = 0
                        print("Settings reloaded: restarting sequence due to shuffle/range change.")
                    else:
                        sample_idx = min(sample_idx, len(selected_images) - 1)

                    print(
                        "Settings reloaded: "
                        f"display_time={display_time_seconds * 1000:.0f}ms, "
                        f"frame_time={frame_time_seconds * 1000:.0f}ms, "
                        f"size={width_px}x{height_px}, "
                        f"padding={padding_px}px, "
                        f"movement_distance={movement_distance_px}px, "
                        f"movement_mode={movement_mode}, "
                        f"shuffle={shuffle_samples}, "
                        f"sample_range={sample_range if sample_range is not None else 'all'}, "
                        f"window_pos=({window_x}, {window_y}), "
                        f"digit_filter={selected_desc}"
                    )

            last_settings_mtime_ns = current_mtime_ns

        if selected_images.size == 0:
            empty_frame = make_frame(
                image=None,
                label=None,
                sample_idx=0,
                total_samples=1,
                movement_mode=movement_mode,
                window_width_px=window_width_px,
                window_height_px=window_height_px,
                image_left_px=image_left_px,
                image_top_px=image_top_px,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                offset_x=0,
                offset_y=0,
                title_band_px=title_band_px,
                message="No images for current digit filter. Edit YAML to continue.",
            )
            cv2.imshow(window_name, empty_frame)

            key = cv2.waitKey(100) & 0xFF
            if key == 27 or key == ord("q"):
                cv2.destroyAllWindows()
                print("Display window closed by user. Stopping.")
                return
            continue

        image = selected_images[sample_idx]
        label = int(selected_labels[sample_idx])
        sample_start = time.monotonic()
        last_sample_idx = -1
        frame_idx = 0

        while True:
            elapsed = time.monotonic() - sample_start
            if elapsed >= display_time_seconds:
                break

            offset_x, offset_y = movement_positions[frame_idx % len(movement_positions)]

            if last_sample_idx != sample_idx:
                digit_time = time.time()
                print("Digit shown:", label, "Time:", digit_time)
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(f"{label},{digit_time}\n")
                last_sample_idx = sample_idx

            frame = make_frame(
                image=image,
                label=label,
                sample_idx=sample_idx,
                total_samples=len(selected_images),
                movement_mode=movement_mode,
                window_width_px=window_width_px,
                window_height_px=window_height_px,
                image_left_px=image_left_px,
                image_top_px=image_top_px,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                offset_x=offset_x,
                offset_y=offset_y,
                title_band_px=title_band_px,
            )
            cv2.imshow(window_name, frame)

            wait_ms = max(1, int(min(frame_time_seconds, max(0.0, display_time_seconds - elapsed)) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27 or key == ord("q"):
                cv2.destroyAllWindows()
                print("Display window closed by user. Stopping.")
                return

            frame_idx += 1

        # Show black screen for silence period
        black_frame = np.zeros((window_height_px, window_width_px, 3), dtype=np.uint8)
        cv2.imshow(window_name, black_frame)
        cv2.waitKey(silence_ms)
        sample_idx += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

