# Py Image Processing Histogram

## Overview

**Py Image Processing Histogram** is a Python-based application that allows users to perform various image transformations using histogram equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE). The application provides a graphical user interface (GUI) built with Tkinter, making it easy to visualize the effects of different image processing techniques.

## Features

- **Histogram Equalization**: Improve the contrast of images by redistributing the intensity levels.
- **CLAHE**: Apply adaptive histogram equalization with adjustable clip limit and grid size.
- **Image Display**: Load and display images from a specified folder.
- **Dynamic GUI**: Real-time updates of image transformations based on user input.

## Requirements

To run this project, ensure you have the following installed:

- Python 3.x
- OpenCV
- NumPy
- Pillow
- Tkinter

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GarikHarutyunyan/py-image-processing-histogram.git

2. **Navigate to the Project Directory:**
   ```bash
   cd py-image-processing-histogram

3. **Create a Virtual Environment:**
   ```bash
   python -m venv venv

4. **Activate the Virtual Environment:**
   
   *On Windows:*
   ```bash
   venv\Scripts\activate
   ```

    *On macOS and Linux:*
   ```bash
   source venv/Scripts/activate

5. **Install Dependencies: Install the required packages using the requirements.txt file:**
   ```bash
   pip install -r requirements.txt

6. **Usage:**
   ```bash
   python main.py
