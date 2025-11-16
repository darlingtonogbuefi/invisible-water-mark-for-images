import sys
import cv2
import numpy as np
import json
import os

# ---------------- Helper Functions ----------------

def text_to_bits(text):
    """Convert string to bits."""
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    """Convert bits back to string until EOF marker."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if byte == "11111110":  # EOF marker
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def save_metadata(output_path, metadata):
    """Save metadata JSON alongside image."""
    meta_path = output_path + ".meta"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

def load_metadata(image_path):
    """Load metadata JSON if it exists."""
    meta_path = image_path + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None

# ---------------- Encoding ----------------

def encode(image_path, output_path, watermark_text, spread=20, delta=2):
    """
    Encode watermark into a JPEG image using DCT.

    Args:
        image_path: Path to input image.
        output_path: Path to save watermarked image.
        watermark_text: String to embed.
        spread: Number of consecutive blocks per bit (increased to 20 for robustness).
        delta: Coefficient change magnitude for embedding.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    bits = text_to_bits(watermark_text) + "11111110"  # EOF marker
    total_bits = len(bits)
    idx = 0  # block counter

    h, w = y.shape

    # Loop through 8x8 blocks
    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            if idx >= total_bits * spread:
                break

            bit_idx = idx // spread
            bit = bits[bit_idx]

            block = y[i:i+8, j:j+8].astype(np.float32)
            dct_block = cv2.dct(block)

            coeff = dct_block[4,3]
            if bit == '1':
                coeff = abs(coeff) + delta
            else:
                coeff = -abs(coeff) - delta
            dct_block[4,3] = coeff

            y[i:i+8, j:j+8] = cv2.idct(dct_block)
            idx += 1

    watermarked = cv2.merge([y, cr, cb])
    watermarked = cv2.cvtColor(watermarked.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, watermarked, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # Save original shape metadata
    metadata = {"original_shape": [h, w], "spread": spread}
    save_metadata(output_path, metadata)

    print(f"Watermark encoded into {output_path}")

# ---------------- Decoding ----------------

def decode(image_path, threshold=1.0):
    """
    Decode watermark from JPEG image, robust to resizing.

    Args:
        image_path: Path to watermarked image.
        threshold: Minimum coefficient magnitude to detect bit.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(ycc)

    # Load metadata
    metadata = load_metadata(image_path)
    if metadata and "original_shape" in metadata:
        h_orig, w_orig = metadata["original_shape"]
        spread = metadata.get("spread", 20)
        y = cv2.resize(y, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    else:
        spread = 20  # default if metadata missing

    bits_list = []
    h, w = y.shape

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = y[i:i+8, j:j+8].astype(np.float32)
            dct_block = cv2.dct(block)
            coeff = dct_block[4,3]

            bit = '1' if coeff >= threshold else '0'
            bits_list.append(bit)

    # Majority vote over spread
    recovered_bits = ""
    for i in range(0, len(bits_list), spread):
        chunk = bits_list[i:i+spread]
        if not chunk:
            break
        ones = chunk.count('1')
        zeros = chunk.count('0')
        recovered_bits += '1' if ones > zeros else '0'

        if recovered_bits[-8:] == "11111110":
            break

    watermark = bits_to_text(recovered_bits)
    print("Decoded watermark:", watermark)

# ---------------- CLI Interface ----------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python watermark.py encode <input> <output> <text>")
        print("  python watermark.py decode <image>")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "encode":
        if len(sys.argv) != 5:
            print("Usage: python watermark.py encode <input> <output> <text>")
            sys.exit(1)
        _, _, input_file, output_file, text = sys.argv
        encode(input_file, output_file, text)

    elif mode == "decode":
        if len(sys.argv) != 3:
            print("Usage: python watermark.py decode <image>")
            sys.exit(1)
        _, _, input_file = sys.argv
        decode(input_file)

    else:
        print("Unknown command:", mode)
