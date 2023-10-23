import logging
import os
from typing import Any, Tuple

import cv2
import ezdxf
import numpy as np
from ezdxf import units
from skimage.morphology import medial_axis, skeletonize

mouse_x, mouse_y = None, None


def load_texture(path: str) -> np.ndarray:
    """Load a file, convert to binary and invert"""
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh


def apply_transform(n: int, input_img: np.ndarray) -> Tuple[str, np.ndarray | None]:
    """Switch case dictionary to apply transforms[n] to the input image"""
    transforms: dict[int, Tuple[str, Any]]
    transforms = {
        0: ("skeletonize", np.asarray(skeletonize(input_img), dtype=np.uint8) * 255),
        1: ("medial_axis", np.asarray(medial_axis(input_img), dtype=np.uint8) * 255),
        2: ("canny", cv2.Canny(input_img, 100, 200)),
    }
    name, transformed_img = transforms.get(n % len(transforms), ('Transform not found', None))
    return name, transformed_img


def check_waitkey(key: int, char: str) -> bool:
    """Helper function for checking opencv display window button presses"""
    return True if key in [ord(char.lower()), ord(char.upper())] else False


def generate_dxf(img: np.ndarray) -> None:
    """Detect contours in the raster image, then generate a DXF and export"""
    # Extract contours from raster image
    contours_pixel, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    scaling = np.array([TARGET_HEIGHT, TARGET_WIDTH]) / img.shape[:2]
    contours_mm = [np.squeeze(cnt, axis=1).astype(np.float32) * scaling for cnt in contours_pixel]

    # Filter out duplicates by testing if centroid means 1% of each other
    contour_means = [ctr.mean() for ctr in contours_mm]
    contours_filtered = [contours_mm[0]]
    for n, ctr in enumerate(contours_mm):
        if n == 0:
            continue
        elif abs((contour_means[n] - contour_means[n-1]) / contour_means[n]) > 0.01:
            contours_filtered.append(contours_mm[n])
    logging.info(
        f"{len(contours_filtered)} contours found ({len(contours_mm) - len(contours_filtered)} duplicates removed)")

    # Generate DXF
    dwg = ezdxf.new('R2010')
    dwg.units = units.MM
    msp = dwg.modelspace()
    dwg.layers.new(name='lines', dxfattribs={'color': 1})
    for ctr in contours_mm:
        # msp.add_spline(ctr, dxfattribs={"layer": "lines", "lineweight": 1})
        msp.add_lwpolyline(ctr, dxfattribs={"layer": "lines", "lineweight": 1})

    # Save DXF
    output_dir = os.path.join(os.getcwd(), 'vectors')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    fname = f"{len(os.listdir(output_dir)):03d}.dxf"
    dwg.saveas(os.path.join(output_dir, fname))
    logging.info(f"{fname} saved")


def draw_bounding(display_img: np.ndarray) -> Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Draws the target bounding rectangle. Also returns the top-left and bottom-right points of the rectangle"""
    # Calculate scale
    scl = np.min(np.array([CANVAS_WIDTH, CANVAS_HEIGHT]) / np.array([TARGET_WIDTH, TARGET_HEIGHT])) * BBOX_MAX
    cx, cy = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2

    # Draw rectangle
    p1 = int(cx - scl * TARGET_WIDTH / 2), int(cy - scl * TARGET_HEIGHT / 2)
    p2 = int(cx + scl * TARGET_WIDTH / 2), int(cy + scl * TARGET_HEIGHT / 2)
    cv2.rectangle(display_img, p1, p2, (0, 0, 255), 2)

    # Add labels
    l1 = (cx - 20, p1[1] - 2)
    l2 = (p2[0], cy)
    cv2.putText(display_img, f"{TARGET_WIDTH}mm", l1, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv2.putText(display_img, f"{TARGET_HEIGHT}mm", l2, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    return display_img, (p1, p2)


def calculate_tiling(texture: np.ndarray, texture_scale: float) -> Tuple[float, float]:
    """For a given texture_scale, calculate the how many times the scaled texture should be tiled in the u and v axes to fill the target rect"""
    base_ratios = texture.shape[:2] / np.array([TARGET_HEIGHT, TARGET_WIDTH])
    scaled_ratios = base_ratios / texture_scale
    tiling = scaled_ratios / np.max(base_ratios)
    return tiling[1], tiling[0]


def tile_texture(texture: np.ndarray, vu_tiling: Tuple[float, float]) -> np.ndarray:
    """Pad/ wrap  or crop a texture in the v (height) and u (width) axes. Padding is performed in the positive and negative directions on each axis, so the texture remains in the centre"""
    texture_padded = texture
    pad_height, pad_width = texture.shape[:2] * (np.array(vu_tiling) - 1)

    # Wrap or crop in the width direction
    if pad_width % 2 == 1:
        pad_left, pad_right = int(np.floor(pad_width / 2)), int(np.ceil(pad_width / 2))
    else:
        pad_left = pad_right = int(pad_width / 2)
    if pad_width > 0:
        texture_padded = np.pad(texture_padded, ((0, 0), (pad_left, pad_right)), 'wrap')
    elif pad_width < 0:
        texture_padded = texture_padded[:, -pad_left: pad_right]
    else:
        pass

    # Wrap or crop in the height direction
    if pad_height % 2 == 1:
        pad_top, pad_bottom = int(np.floor(pad_height / 2)), int(np.ceil(pad_height / 2))
    else:
        pad_top = pad_bottom = int(pad_height / 2)
    if pad_height > 0:
        texture_padded = np.pad(texture_padded, ((pad_top, pad_bottom), (0, 0)), 'wrap')
    elif pad_height < 0:
        texture_padded = texture_padded[-pad_top: pad_bottom, :]
    else:
        pass

    return texture_padded


def draw_bit_profile(img: np.ndarray, mouse_xpos: int, mouse_ypos: int) -> None:
    """Draw a circle representing the bit diameter at the mouse position. Use to help dial in the correct scaling"""
    scl = np.min(np.array([CANVAS_WIDTH, CANVAS_HEIGHT]) / np.array([TARGET_WIDTH, TARGET_HEIGHT])) * BBOX_MAX
    cv2.circle(img, (mouse_xpos, mouse_ypos), int(scl * BIT_DIAMETER / 2), (255, 0, 0), -1)


def draw_text(img: np.ndarray, texture_path: str, texture_scale: float, transform_name: str) -> None:
    """Add text information to the display image"""
    font, font_size, font_colour = cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0)
    cv2.putText(img, f"{texture_path = }", (30, 20), font, font_size, font_colour)
    cv2.putText(img, f"{texture_scale = }", (30, 40), font, font_size, font_colour)
    cv2.putText(img, f"{transform_name = }", (30, 60), font, font_size, font_colour)
    cv2.putText(img, f"W/S : Zoom in/out", (30, CANVAS_HEIGHT - 60), font, font_size, font_colour)
    cv2.putText(img, f"R/F : Scroll through textures", (30, CANVAS_HEIGHT - 40), font, font_size, font_colour)
    cv2.putText(img, f"T/G : Scroll through transforms", (30, CANVAS_HEIGHT - 20), font, font_size, font_colour)
    cv2.putText(img, f"X : Export", (CANVAS_WIDTH // 2 + 30, CANVAS_HEIGHT - 60), font, font_size, font_colour)
    cv2.putText(img, f"Q : Quit", (CANVAS_WIDTH // 2 + 30, CANVAS_HEIGHT - 40), font, font_size, font_colour)


def on_mouse(event, x: int, y: int, flags, param) -> None:
    """Callback for capturing mouse position"""
    global mouse_x
    global mouse_y
    mouse_x, mouse_y = x, y


def main():

    texture_id = 0
    transform_id = 0
    texture_scale = 1.0

    input_dir = os.path.join(os.getcwd(), 'textures')
    texture_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)]
    texture_raw = load_texture(texture_paths[texture_id])

    while True:
        try:

            # Draw canvas and bounding rect
            canvas = np.zeros([CANVAS_HEIGHT, CANVAS_WIDTH, 3])
            canvas, (p1, p2) = draw_bounding(canvas)

            # Calculate and apply tiling
            tiling = calculate_tiling(texture_raw, texture_scale)
            texture_full = tile_texture(texture_raw, tiling)

            # Downscale and draw the transformed texture
            texture_display = cv2.resize(texture_full, [p2[0] - p1[0], p2[1] - p1[1]])
            tf_name, tf_display = apply_transform(transform_id, texture_display)
            if tf_display is not None:
                canvas[p1[1]:p2[1], p1[0]: p2[0]] = cv2.cvtColor(tf_display, cv2.COLOR_GRAY2BGR)

            # Display
            draw_text(canvas, os.listdir(input_dir)[texture_id], texture_scale, tf_name)
            if mouse_x and mouse_y:
                draw_bit_profile(canvas, mouse_x, mouse_y)
            cv2.imshow('output', canvas)
            cv2.setMouseCallback('output', on_mouse)
            k = cv2.waitKey(1)

            # Switch texture
            if check_waitkey(k, 'r'):
                texture_id += 1
                texture_raw = load_texture(texture_paths[texture_id % len(texture_paths)])
            if check_waitkey(k, 'f'):
                texture_id -= 1
                texture_raw = load_texture(texture_paths[texture_id % len(texture_paths)])

            # Switch transform
            if check_waitkey(k, 't'):
                transform_id += 1
            if check_waitkey(k, 'g'):
                transform_id -= 1

            # Scale texture
            if check_waitkey(k, 'w'):
                texture_scale *= 1.1
            if check_waitkey(k, 's'):
                texture_scale /= 1.1

            # Export DXF
            if check_waitkey(k, 'x'):
                # Upscale and re-threshold the image
                blur = cv2.GaussianBlur(texture_full, [13, 13], 0)
                upscaled = cv2.resize(blur, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                _, upscaled_thresh = cv2.threshold(upscaled, 127, 255, cv2.THRESH_BINARY)
                _, tf_full = apply_transform(transform_id, upscaled_thresh)
                if tf_full is not None:
                    generate_dxf(tf_full)

            # Close
            if check_waitkey(k, 'q'):
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    CANVAS_WIDTH, CANVAS_HEIGHT = 800, 800  # pixels
    BBOX_MAX = 0.8  # Max size of the bounding rectangle reletave to the canvas

    TARGET_WIDTH, TARGET_HEIGHT = 300, 300  # mm
    BIT_DIAMETER = 30  # mm
    main()
