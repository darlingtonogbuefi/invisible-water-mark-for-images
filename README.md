# Invisible Watermark for Images

A Python tool to **embed and extract invisible text watermarks** in JPEG images using DCT (Discrete Cosine Transform). The watermarking method is robust to resizing and minor image manipulations.

---

## Features

- Encode secret text into an image without visible changes.
- Decode text from a watermarked image.
- Supports JPEG images.
- Stores metadata for accurate decoding.

---

## Installation

1. Clone the repository:

git clone https://github.com/darlingtonogbuefi/invisible-water-mark-for-images.git
cd invisible-water-mark-for-images


Install dependencies:

2. pip install -r requirements.txt

3. Usage

Encode a watermark
Embed a secret text into an image:
python watermark.py encode <input_image> <output_image> "<watermark_text>"

Example:
python watermark.py encode input.jpeg output.jpeg "MySecret"

Decode a watermark
Retrieve the hidden text from a watermarked image:
python watermark.py decode <watermarked_image>

Example:
python watermark.py decode AWS_EKS_ElasticCloud_wm.jpeg