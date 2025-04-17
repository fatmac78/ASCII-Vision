#!/usr/bin/env python

# Alex Eidt

import tkinter as tk
import string
import imageio
import numpy as np
import numexpr as ne
import keyboard
from PIL import Image, ImageTk, ImageFont, ImageDraw


# Mirror image stream along vertical axis.
MIRROR = True
# Video Stream to use.
STREAM = "<video0>"
# Background color of the ASCII stream.
BACKGROUND_COLOR = "white"
# Font color used in the ASCII stream. Make sure there's some contrast between the two.
FONT_COLOR = "black"
# Font size to use with colored/grayscaled ASCII.
FONTSIZE = 12
# Boldness to use with colored/grayscaled ASCII.
BOLDNESS = 1
# Factor to divide image height and width by. 1 For for original size, 2 for half size, etc...
FACTOR = 1
# Characters to use in ASCII.
CHARS = "@%#*+=-:. "

# Font to use in ASCII Graphics.
FONT = "cour.ttf"


ASCII = 0
FILTER = 0
BLOCKS = 0
TEXT = 0
MONO = 0
MIRROR = 1


def get_font_maps(fontsize, boldness, chars):
    """
    Returns a list of font bitmaps.
    Parameters
        fontsize    - Font size to use for ASCII characters
        boldness    - Stroke size to use when drawing ASCII characters
        chars       - ASCII characters to use in media
    Returns
        List of font bitmaps corresponding to the order of characters in CHARS
    """
    fonts = []
    widths, heights = set(), set()
    font = ImageFont.truetype(FONT, size=fontsize)
    for char in chars:
        w, h = font.getsize(char)
        widths.add(w)
        heights.add(h)
        image = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, -(fontsize // 6)),
            char,
            fill=(0, 0, 0),
            font=font,
            stroke_width=boldness,
        )
        bitmap = np.array(image, dtype=np.uint8)
        fonts.append(255 - bitmap)

    fonts = list(map(lambda x: x[: min(heights), : min(widths)], fonts))
    return np.array(sorted(fonts, key=lambda x: x.sum(), reverse=True))


def update():
    """
    Update settings based on user input.
    """
    global ASCII, FILTER, BLOCKS, TEXT, MONO

    if keyboard.is_pressed("shift+a"):  # ASCII Mode.
        ASCII = 0
    elif keyboard.is_pressed("a"):
        ASCII = 1
    if keyboard.is_pressed("shift+t"):  # Text Mode.
        TEXT = 0
    elif keyboard.is_pressed("t"):
        TEXT = 1
    if keyboard.is_pressed("shift+m"):  # Monochromatic Mode.
        MONO = 0
    elif keyboard.is_pressed("m"):
        MONO = 1

    if keyboard.is_pressed("o"):  # Outline Filter.
        FILTER = 1
    elif keyboard.is_pressed("s"):  # Sobel Filter.
        FILTER = 2
    elif keyboard.is_pressed("space"):  # No Filter.
        FILTER = 0

    for i in range(10):
        if keyboard.is_pressed(str(i)):
            BLOCKS = i
            break


def tile_tuples(w, h):
    """
    Return tile sizes for resizing ASCII Images.
    """
    result = lambda x: [i for i in range(2, x) if x % i == 0]
    return list(zip(result(w), result(h)))


def convolve(frame, kernel):
    """
    Peform a 2D image convolution on the given frame with the given kernel.
    """
    height, width = frame.shape
    kernel_height, kernel_width = kernel.shape
    # assert kh == kw
    output = np.pad(frame, kernel_height // 2, mode="edge")

    output_shape = kernel.shape + tuple(np.subtract(output.shape, kernel.shape) + 1)
    strides = output.strides + output.strides

    return np.einsum(
        "ij,ijkl->kl",
        kernel,
        np.lib.stride_tricks.as_strided(output, output_shape, strides),
    )


def main():
    # All ASCII characters used in the images sorted by pixel density.
    chars = np.array([c for c in string.printable if c in CHARS])
    font_maps = [get_font_maps(FONTSIZE, BOLDNESS, chars)]
    for fontsize in [5, 10, 15, 20, 30, 45, 60, 85, 100]:
        font_maps.append(get_font_maps(fontsize, BOLDNESS, chars))

    # Set up window.
    root = tk.Tk()
    root.title("ASCII Streamer")
    mainframe = tk.Frame()
    image_label = tk.Label(mainframe, borderwidth=5, relief="solid")
    ascii_label = tk.Label(
        mainframe,
        font=("courier", 2),
        fg=FONT_COLOR,
        bg=BACKGROUND_COLOR,
        borderwidth=5,
        relief="solid",
    )
    mainframe.pack(side=tk.LEFT, expand=tk.YES, padx=10)
    root.protocol("WM_DELETE_WINDOW", lambda: (video.close(), root.destroy()))

    # Get image stream from webcam or other source and begin streaming.
    video = imageio.get_reader(STREAM)
    w, h = video.get_meta_data()["source_size"]

    tiles = tile_tuples(w, h)

    def stream():
        image = video.get_next_data()
        # Update settings based on pressed keys.
        update()

        h, w, c = image.shape
        # Text image is larger than regular, so multiply scaling factor by 2 if Text mode is on.
        size = FACTOR * 2 if TEXT else FACTOR
        h //= size
        w //= size

        # Resize Image.
        image = image[::size, ::size]
        if TEXT:  # Grayscale Image.
            image = (image * np.array([0.299, 0.587, 0.114])).sum(
                axis=2, dtype=np.uint8
            )
        if MIRROR:  # Mirror Image along vertical axis.
            image = image[:, ::-1]

        # Tile Image into dw x dh blocks for resized ASCII streams.
        if BLOCKS > 0 and TEXT:
            dw, dh = tiles[min(BLOCKS, len(tiles) - 1)]
            image = (
                np.add.reduceat(
                    np.add.reduceat(image.astype(np.uint32), np.arange(0, h, dh), axis=0),
                    np.arange(0, w, dw),
                    axis=1,
                )
                / (dw * dh)
            ).astype(np.uint8)
            h, w = image.shape

        # Apply image convolutions to stream.
        if FILTER > 0 and TEXT:
            if FILTER == 1:  # Outline Kernel.
                image = convolve(
                    image, np.array([[-1, -1, -1], [-1, -8, -1], [-1, -1, -1]])
                ).astype(np.uint8)
            elif FILTER == 2:  # Sobel Kernel.
                gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
                gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                image = np.hypot(convolve(image, gx), convolve(image, gy)).astype(
                    np.uint8
                )

        if ASCII and not TEXT:
            fh, fw = font_maps[BLOCKS][0].shape[:2]
            frame = image[::fh, ::fw]
            nh, nw = frame.shape[:2]

            if not MONO:
                colors = (255 - frame).repeat(fw, 1).repeat(fh, 0)

            grayscaled = (
                (frame * np.array([3, 4, 1])).sum(axis=2, dtype=np.uint32).ravel()
            )

            grayscaled *= len(chars)
            grayscaled >>= 11

            # Create a new list with each font bitmap based on the grayscale value
            image = (
                font_maps[BLOCKS][grayscaled]
                .reshape((nh, nw, fh, fw, 3))
                .transpose(0, 2, 1, 3, 4)
                .ravel()
                .reshape((nh * fh, nw * fw, 3))
            )
            if MONO:
                ne.evaluate("255 - image", out=image, casting="unsafe")
                image = image.astype(np.uint8)
            else:
                image = image[:h, :w]
                colors = colors[:h, :w].astype(np.uint16)
                image = (255 - (image * colors) // 255).astype(np.uint8)

        # If ASCII mode is on convert frame to ascii and display, otherwise display video stream.
        if TEXT:
            image = image[[i for i in range(h) if i % 4]]
            image = image.astype(np.uint32)
            image *= len(chars)
            image >>= 8
            image_label.pack_forget()
            ascii_label.pack()
            # Update label with new ASCII image.
            ascii_label.config(
                text="\n".join("".join(x) for x in chars[image]),
                font=("courier", (BLOCKS * 4) + 2),
            )
            ascii_label.after(1, stream)
        else:
            ascii_label.pack_forget()
            image_label.pack()
            frame_image = ImageTk.PhotoImage(Image.fromarray(image))
            image_label.config(image=frame_image)
            image_label.image = frame_image
            image_label.after(1, stream)

    stream()
    root.state("zoomed")
    root.mainloop()


if __name__ == "__main__":
    main()
