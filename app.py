#!/usr/bin/env python

# Alex Eidt

import tkinter as tk
import string
import imageio
import numpy as np
import numexpr as ne
from PIL import Image, ImageTk, ImageFont, ImageDraw


# Mirror image stream along vertical axis.
MIRROR = True
# Video Stream to use.
STREAM = "<video0>"
# Background color of the ASCII stream.
BACKGROUND_COLOR = "black"
# Font color used in the ASCII stream. Make sure there's some contrast between the two.
FONT_COLOR = "green"
# Factor to divide image height and width by. 1 For for original size, 2 for half size, etc...
FACTOR = 1
# Characters to use in ASCII.
CHARS = "@%#*+=-:. "
# Sobel filter strength (new variable)
SOBEL_STRENGTH = 0.2
# Font to use in ASCII Graphics (missing variable that's causing the error)
FONT = "cour.ttf"  # Common monospace font, or you can use a system font path

FILTER = 2
BLOCKS = 0
MIRROR = 1
INVERT = 1  # 0 = Off, 1 = On


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

    # Set up window.
    root = tk.Tk()
    root.title("ASCII Streamer")

    # Configure the root window to use the full space
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Change from Frame to Frame with configured weights
    mainframe = tk.Frame(root, bg="black")
    mainframe.grid(row=0, column=0, sticky="nsew")  # Use grid instead of pack
    mainframe.grid_rowconfigure(0, weight=1)
    mainframe.grid_columnconfigure(0, weight=1)

    # Configure image_label to fill the entire frame
    image_label = tk.Label(mainframe, borderwidth=0, bg="black")
    image_label.grid(row=0, column=0, sticky="nsew")  # Use grid instead of pack

    # Configure ascii_label to fill the entire frame
    ascii_label = tk.Label(
        mainframe,
        font=("courier", 2),
        fg=FONT_COLOR,
        bg=BACKGROUND_COLOR,
        borderwidth=0,
    )
    ascii_label.grid(row=0, column=0, sticky="nsew")  # Use grid instead of pack

    root.protocol("WM_DELETE_WINDOW", lambda: (video.close(), root.destroy()))

    # Get image stream from webcam or other source and begin streaming.
    video = imageio.get_reader(STREAM)
    w, h = video.get_meta_data()["source_size"]

    tiles = tile_tuples(w, h)

    # Initialize BLOCKS and calculate initial font size
    global BLOCKS
    initial_font_size = (BLOCKS * 4) + 2  # This is apparently the current formula
    
    # Print initial font size
    print(f"Initial BLOCKS: {BLOCKS}, Font size: {initial_font_size}")

    # Define key event handlers
    def on_key_press(event):
        global FILTER, BLOCKS, MIRROR, INVERT, SOBEL_STRENGTH
        
        key = event.keysym
        print(f"Key pressed: {key}, keysym: {event.keysym}, keycode: {event.keycode}")
        
        # Handle number keys specifically
        if key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            old_blocks = BLOCKS
            old_font_size = (old_blocks * 4) + 2
            
            # For number keys 1-9, set BLOCKS
            if key != '0':
                BLOCKS = int(key)
            else:
                BLOCKS = 0
                
            # Calculate new font size - modify this formula to make sizes smaller
            # Option 1: Divide by 2 to make sizes smaller
            new_font_size = (BLOCKS * 2) + 2
            
            print(f"BLOCKS changed: {old_blocks} → {BLOCKS}")
            print(f"Font size changed: {old_font_size} → {new_font_size}")
            return
            
        # Rest of key handling
        if key == 'i':  # Add this for inversion
            INVERT = not INVERT
            print(f"Invert Mode: {'ON' if INVERT else 'OFF'}")
            
        # Filter settings
        elif key == 's':
            FILTER = 1  # Change from 2 to 1 since we only have one filter now
            print("Filter: Sobel")
        elif key == 'space':
            FILTER = 0
            print("Filter: None")
        
        # Handle Sobel strength adjustments
        if key == 'e':  # Increase Sobel strength
            SOBEL_STRENGTH = min(10.0, SOBEL_STRENGTH + 0.1)
            SOBEL_STRENGTH = round(SOBEL_STRENGTH, 1)  # Round to 1 decimal place
            print(f"Sobel strength INCREASED to: {SOBEL_STRENGTH}")
        elif key == 'd':  # Decrease Sobel strength
            SOBEL_STRENGTH = max(0.1, SOBEL_STRENGTH - 0.1)
            SOBEL_STRENGTH = round(SOBEL_STRENGTH, 1)  # Round to 1 decimal place
            print(f"Sobel strength DECREASED to: {SOBEL_STRENGTH}")
    
    # Bind each key individually
    root.bind("<KeyPress-s>", on_key_press)
    root.bind("<KeyPress-i>", on_key_press)  # Bind inversion key
    root.bind("<KeyPress-e>", on_key_press)  # Bind increase Sobel strength key
    root.bind("<KeyPress-d>", on_key_press)  # Bind decrease Sobel strength key
    root.bind("<space>", on_key_press)

    # Bind number keys individually
    for i in range(10):
        root.bind(f"<KeyPress-{i}>", on_key_press)

    # Also bind to general key press for debugging
    root.bind("<KeyPress>", on_key_press)
    
    # Make sure the window has focus to receive key events
    root.focus_force()

    def stream():
        global FILTER, BLOCKS, MIRROR, INVERT

        image = video.get_next_data()

        h, w, c = image.shape
        size = FACTOR * 2 
        h //= size
        w //= size

        # Resize Image.
        image = image[::size, ::size]
        image = (image * np.array([0.299, 0.587, 0.114])).sum(
                axis=2, dtype=np.uint8
            )
        if MIRROR:  # Mirror Image along vertical axis.
            image = image[:, ::-1]

        # Tile Image into dw x dh blocks for resized ASCII streams.
        if BLOCKS > 0:
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
        if FILTER > 0:
            # Sobel Kernel
            gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) * SOBEL_STRENGTH
            gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * SOBEL_STRENGTH
            image = np.hypot(convolve(image, gx), convolve(image, gy)).astype(
                np.uint8
            )

        # Apply inversion if enabled
        if INVERT:
            image = 255 - image
       
        image = image[[i for i in range(h) if i % 4]]
        image = image.astype(np.uint32)
        image *= len(chars)
        image >>= 8
        image_label.grid_remove()
        ascii_label.grid()
        # Update label with new ASCII image.
        font_size = (BLOCKS * 2) + 2  # Changed from (BLOCKS * 4) + 2
        ascii_label.config(
            text="\n".join("".join(x) for x in chars[image]),
            font=("courier", font_size),  # Use the new formula
        )
        ascii_label.after(1, stream)

    stream()
    try:
        root.attributes('-zoomed', True)  # Works on many Linux systems
    except Exception:
        # Fallback method - set window size to match screen
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.geometry(f"{width}x{height}+0+0")
    root.mainloop()


if __name__ == "__main__":
    main()
