import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
import os
from transformations import histogram_equalization, clahe

# Folder containing images
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'images')
EQUALIZE_DEFAULT_VALUE = 1
FIRST_CLIP_LIMIT_DEFAULT_VALUE = 2.0
FIRST_GRID_SIZE_DEFAULT_VALUE = 8
SECOND_CLIP_LIMIT_DEFAULT_VALUE = 4.0
SECOND_GRID_SIZE_DEFAULT_VALUE = 16

def resize_image(image, width=200):
    height = int(image.shape[0] * (width / image.shape[1]))
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def display_images(images, titles, options):
    global first_clahe_label, second_clahe_label # Make labels global to update later
    for widget in frame.winfo_children():
        widget.destroy()

    for i in range(len(images)):
        resized_image = resize_image(images[i])
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        title_label = Label(frame, text=titles[i])
        title_label.grid(row=(i // 7) * 2, column=i % 7, padx=5, pady=(0, 5))

        label = Label(frame, image=image_tk)
        label.image = image_tk
        label.grid(row=(i // 7) * 2 + 1, column=i % 7, padx=5, pady=(0, 5))
        
        get_option = options[i];
        if get_option is not None:
            get_option().grid(row=(i // 7) * 2+2, column=i % 7, padx=5, pady=(0, 5))

        # Keep a reference  labels
        if titles[i] == 'CLAHE (Clip=2.0, Grid=8x8)':
            first_clahe_label = label
        elif titles[i] == 'CLAHE (Clip=4.0, Grid=16x16)':
            second_clahe_label = label


def apply_transformations(selected_image):
    global original_image, frame
    if selected_image is None:
        return

    def get_clahe_first_option():        
        global first_clip_limit, first_grid_size
        
        options_frame = tk.Frame(frame)
        first_clip_limit = tk.Scale(options_frame, from_=1.0, to=3.0, orient='horizontal', label='Clip limit', command=on_first_clip_limit_change)
        first_clip_limit.set(FIRST_CLIP_LIMIT_DEFAULT_VALUE)        
        first_clip_limit.grid(column=1, row=0)
        first_grid_size = tk.Scale(options_frame, from_=2, to=20, orient='horizontal', label='Grid Size', command=on_first_grid_size_change)
        first_grid_size.set(FIRST_GRID_SIZE_DEFAULT_VALUE)
        first_grid_size.grid(column=2, row=0)
        return options_frame
    
    def get_clahe_second_option():        
        global second_clip_limit, second_grid_size
        
        options_frame = tk.Frame(frame)
        second_clip_limit = tk.Scale(options_frame, from_=3.0, to=15.0, orient='horizontal', label='Clip limit', command=on_second_clip_limit_change)
        second_clip_limit.set(SECOND_CLIP_LIMIT_DEFAULT_VALUE)        
        second_clip_limit.grid(column=1, row=0)
        second_grid_size = tk.Scale(options_frame, from_=2, to=20, orient='horizontal', label='Grid Size', command=on_second_grid_size_change)
        second_grid_size.set(SECOND_GRID_SIZE_DEFAULT_VALUE)
        second_grid_size.grid(column=2, row=0)
        return options_frame

    original_image = selected_image

    equalized_image = histogram_equalization(selected_image)
    clahe_image_1 = clahe(selected_image, clip_limit=FIRST_CLIP_LIMIT_DEFAULT_VALUE, grid_size=FIRST_GRID_SIZE_DEFAULT_VALUE)
    clahe_image_2 = clahe(selected_image, clip_limit=SECOND_CLIP_LIMIT_DEFAULT_VALUE, grid_size=SECOND_GRID_SIZE_DEFAULT_VALUE)
    
    images = [selected_image, equalized_image, clahe_image_1, clahe_image_2]
    titles = ['Original', 'Histogram Equalization', 'CLAHE (Clip=2.0, Grid=8x8)', 'CLAHE (Clip=4.0, Grid=16x16)']
    options = [None, None, get_clahe_first_option, get_clahe_second_option]

    display_images(images, titles, options)

def on_selection_change(event):
    print(event)
    selected_image_name = dropdown_var.get()
    selected_image_path = os.path.join(IMAGE_FOLDER, selected_image_name)
    selected_image = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
    apply_transformations(selected_image)

def on_first_clip_limit_change(_):
    on_first_clahe_change()

def on_first_grid_size_change(_):
    on_first_clahe_change()

def on_first_clahe_change():
    global original_image, first_clahe_label, first_clip_limit, first_grid_size
    if original_image is None:
        return

    clip_limit = first_clip_limit.get()
    grid_size = first_grid_size.get()
    contrast_image = clahe(original_image, clip_limit=clip_limit, grid_size=grid_size)
    resized_image = resize_image(contrast_image)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    first_clahe_label.config(image=image_tk)
    first_clahe_label.image = image_tk

def on_second_clip_limit_change(_):
    on_second_clahe_change()

def on_second_grid_size_change(_):
    on_second_clahe_change()

def on_second_clahe_change():
    global original_image, second_clahe_label, second_clip_limit, second_grid_size
    if original_image is None:
        return

    clip_limit = second_clip_limit.get()
    grid_size = second_grid_size.get()
    contrast_image = clahe(original_image, clip_limit=clip_limit, grid_size=grid_size)
    resized_image = resize_image(contrast_image)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    second_clahe_label.config(image=image_tk)
    second_clahe_label.image = image_tk

# Set up the main Tkinter window
root = tk.Tk()
root.title("Image Transformations")
root.state('zoomed')
# root.attributes('-zoomed', True) # For Linux and macOS


# Set up the dropdown menu
dropdown_var = StringVar()
dropdown_menu = ttk.Combobox(root, textvariable=dropdown_var)
dropdown_menu.bind("<<ComboboxSelected>>", on_selection_change)

image_files = os.listdir(IMAGE_FOLDER)
dropdown_menu['values'] = [f for f in image_files if f.endswith(('.png', '.jpg', '.jpeg'))]
dropdown_menu.pack(pady=10)

if image_files:
    dropdown_var.set(image_files[0])

# Frame for displaying images
frame = tk.Frame(root)
frame.pack(padx=10, pady=200)

original_image = None
binary_label = None
contrast_label = None

# Responsible for initial rendering
on_selection_change(0)

# Start the Tkinter main loop
root.mainloop()
