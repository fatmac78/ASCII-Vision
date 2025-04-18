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
# Font size to use with colored/grayscaled ASCII.
FONTSIZE = 20
# Boldness to use with colored/grayscaled ASCII.
BOLDNESS = 2
# Factor to divide image height and width by. 1 For for original size, 2 for half size, etc...
FACTOR = 1
# Characters to use in ASCII.
CHARS = "@%#*+=-:. "
# Sobel filter strength (new variable)
SOBEL_STRENGTH = 0.2
# Font to use in ASCII Graphics (missing variable that's causing the error)
FONT = "cour.ttf"  # Common monospace font, or you can use a system font path


ASCII = 0
FILTER = 2
BLOCKS = 0
TEXT = 1
MONO = 0
MIRROR = 1
INVERT = 1  # 0 = Off, 1 = On


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
        bitmap = np.asarray(image, dtype=np.uint8)
        fonts.append(255 - bitmap)

    fonts = list(map(lambda x: x[: min(heights), : min(widths)], fonts))
    return np.array(sorted(fonts, key=lambda x: x.sum(), reverse=True))


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
        global ASCII, FILTER, BLOCKS, TEXT, MONO, MIRROR, INVERT, SOBEL_STRENGTH
        
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
            
            # Option 2: Use a more modest progression
            # new_font_size = BLOCKS + 4
            
            # Option 3: Logarithmic progression for more control at small sizes
            # new_font_size = int(math.log2(BLOCKS + 2) * 4) if BLOCKS > 0 else 2
            
            print(f"BLOCKS changed: {old_blocks} → {BLOCKS}")
            print(f"Font size changed: {old_font_size} → {new_font_size}")
            return
            
        # Rest of key handling
        if key == 'a':
            ASCII = not ASCII
            print(f"ASCII Mode: {'ON' if ASCII else 'OFF'}")
        
        elif key == 't':
            TEXT = not TEXT
            print(f"Text Mode: {'ON' if TEXT else 'OFF'}")
        
        elif key == 'm':
            MONO = not MONO
            print(f"Mono Mode: {'ON' if MONO else 'OFF'}")
            
        elif key == 'i':  # Add this for inversion
            INVERT = not INVERT
            print(f"Invert Mode: {'ON' if INVERT else 'OFF'}")
            
        # Filter settings
        elif key == 'o':
            FILTER = 1
            print("Filter: Outline")
        elif key == 's':
            FILTER = 2
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
    root.bind("<KeyPress-a>", on_key_press)
    root.bind("<KeyPress-t>", on_key_press)
    root.bind("<KeyPress-m>", on_key_press)
    root.bind("<KeyPress-o>", on_key_press)
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
        global ASCII, TEXT, FILTER, BLOCKS, MONO, MIRROR, INVERT

        image = video.get_next_data()

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
                gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) * SOBEL_STRENGTH
                gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * SOBEL_STRENGTH
                image = np.hypot(convolve(image, gx), convolve(image, gy)).astype(
                    np.uint8
                )

        # Apply inversion if enabled
        if INVERT:
            image = 255 - image

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
            image_label.grid_remove()
            ascii_label.grid()
            # Update label with new ASCII image.
            font_size = (BLOCKS * 2) + 2  # Changed from (BLOCKS * 4) + 2
            ascii_label.config(
                text="\n".join("".join(x) for x in chars[image]),
                font=("courier", font_size),  # Use the new formula
            )
            ascii_label.after(1, stream)
        else:
            ascii_label.grid_remove()
            image_label.grid()

            # Resize image to fit the window
            window_width = root.winfo_width()
            window_height = root.winfo_height()

            if window_width > 1 and window_height > 1:  # Ensure window has been drawn
                # Convert NumPy array to PIL Image
                pil_image = Image.fromarray(image)

                # Resize to fit window dimensions
                pil_image = pil_image.resize((window_width, window_height), Image.LANCZOS)

                frame_image = ImageTk.PhotoImage(pil_image)
                image_label.config(image=frame_image)
                image_label.image = frame_image

            image_label.after(1, stream)

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
