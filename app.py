import os
import io
import cv2
import struct
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, send_file
)
from werkzeug.utils import secure_filename
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

# --- Config ---
UPLOAD_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
PBKDF2_ITERS = 200_000
SALT_BYTES = 16
NONCE_BYTES = 12

app = Flask(__name__)
app.secret_key = "replace-this-with-a-strong-secret"  # for flash messages

# --- AES helpers (GCM) ---
def aes_encrypt(plaintext: bytes, password: str) -> bytes:
    salt = get_random_bytes(SALT_BYTES)
    key = PBKDF2(password.encode("utf-8"), salt, dkLen=32, count=PBKDF2_ITERS)
    nonce = get_random_bytes(NONCE_BYTES)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return salt + nonce + tag + ciphertext  # packed blob

def aes_decrypt(blob: bytes, password: str) -> bytes:
    if len(blob) < SALT_BYTES + NONCE_BYTES + 16:
        raise ValueError("Encrypted blob too short or corrupted")
    salt = blob[:SALT_BYTES]
    nonce = blob[SALT_BYTES:SALT_BYTES+NONCE_BYTES]
    tag = blob[SALT_BYTES+NONCE_BYTES:SALT_BYTES+NONCE_BYTES+16]
    ciphertext = blob[SALT_BYTES+NONCE_BYTES+16:]
    key = PBKDF2(password.encode("utf-8"), salt, dkLen=32, count=PBKDF2_ITERS)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

# --- Utility functions ---
def allowed_filename(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXT

def embed_bytes_into_image(img_path: str, data: bytes) -> (bytes, str):
    """
    Embed `data` (bytes) into LSBs of image at img_path.
    Adds a 4-byte big-endian length prefix. Returns (stego_image_array, errmsg_or_none)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, "Cannot open input image."

    h, w, ch = img.shape
    capacity_bits = h * w * ch  # one LSB per channel per pixel
    total_bytes_needed = 4 + len(data)  # 4 bytes length prefix + data
    if total_bytes_needed * 8 > capacity_bits:
        return None, f"Payload too large for image. Capacity ≈ {capacity_bits // 8} bytes, need {total_bytes_needed} bytes."

    # Build payload with 4-byte length prefix
    payload = struct.pack(">I", len(data)) + data
    flat = img.flatten()
    # embed bit-by-bit into LSBs
    for i, byte in enumerate(payload):
        for bit in range(8):
            bit_val = (byte >> (7 - bit)) & 1
            flat[i*8 + bit] = (int(flat[i*8 + bit]) & 0xFE) | bit_val
    stego = flat.reshape(img.shape)
    return stego, None

def extract_bytes_from_image(img_path: str) -> (bytes, str):
    """
    Extract bytes from LSBs: reads 4-byte length prefix then that many bytes.
    Returns (data_bytes, errmsg_or_none).
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, "Cannot open image."

    flat = img.flatten().astype(int)
    # first 32 bits => length
    if flat.size < 32:
        return None, "Image too small or corrupted."
    length = 0
    for b in flat[:32] & 1:
        length = (length << 1) | int(b)
    if length <= 0:
        return None, "Invalid length (0) or corrupted payload."
    total_bits = (4 + length) * 8
    if flat.size < total_bits:
        return None, "Image does not contain full payload (truncated)."
    data_bits = flat[:total_bits] & 1
    data_bytes = bytearray()
    for i in range(0, len(data_bits), 8):
        byte = 0
        for bit in data_bits[i:i+8]:
            byte = (byte << 1) | int(bit)
        data_bytes.append(byte)
    return bytes(data_bytes[4:]), None  # skip length prefix in returned data

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/embed", methods=["GET", "POST"])
def embed():
    if request.method == "POST":
        image_file = request.files.get("image")
        secret_text = request.form.get("text", "")
        use_aes = request.form.get("use_aes") == "on"
        password = request.form.get("password", "")

        if not image_file or image_file.filename == "":
            flash("Please upload an image.", "danger")
            return redirect(url_for("embed"))
        if not secret_text:
            flash("Please enter text to hide.", "danger")
            return redirect(url_for("embed"))
        if use_aes and not password:
            flash("Password required when AES is enabled.", "danger")
            return redirect(url_for("embed"))

        if not allowed_filename(image_file.filename):
            flash("Unsupported image format. Use PNG/JPG/BMP.", "danger")
            return redirect(url_for("embed"))

        safe_name = secure_filename(image_file.filename)
        base_name = f"{uuid.uuid4().hex}_{safe_name}"
        cover_path = os.path.join(UPLOAD_FOLDER, base_name)
        image_file.save(cover_path)

        try:
            plaintext_bytes = secret_text.encode("utf-8")
            if use_aes:
                payload = aes_encrypt(plaintext_bytes, password)
            else:
                payload = plaintext_bytes

            stego_arr, err = embed_bytes_into_image(cover_path, payload)
            if err:
                flash(err, "danger")
                # cleanup uploaded file
                try: os.remove(cover_path)
                except: pass
                return redirect(url_for("embed"))

            out_name = f"stego_{base_name}"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, stego_arr)
            flash("Embedding successful. Download your stego image below.", "success")
            return render_template("embed.html", stego_image=out_name)
        except Exception as e:
            flash(f"Error during embedding: {e}", "danger")
            try: os.remove(cover_path)
            except: pass
            return redirect(url_for("embed"))

    return render_template("embed.html", stego_image=None)

@app.route("/extract", methods=["GET", "POST"])
def extract():
    if request.method == "POST":
        image_file = request.files.get("image")
        use_aes = request.form.get("use_aes") == "on"
        password = request.form.get("password", "")

        if not image_file or image_file.filename == "":
            flash("Please upload a stego image.", "danger")
            return redirect(url_for("extract"))
        if use_aes and not password:
            flash("Password required to decrypt AES payload.", "danger")
            return redirect(url_for("extract"))
        if not allowed_filename(image_file.filename):
            flash("Unsupported image format. Use PNG/JPG/BMP.", "danger")
            return redirect(url_for("extract"))

        safe_name = secure_filename(image_file.filename)
        fname = f"uploaded_{uuid.uuid4().hex}_{safe_name}"
        img_path = os.path.join(UPLOAD_FOLDER, fname)
        image_file.save(img_path)

        try:
            extracted_bytes, err = extract_bytes_from_image(img_path)
            if err:
                flash(err, "danger")
                try: os.remove(img_path)
                except: pass
                return redirect(url_for("extract"))

            if use_aes:
                try:
                    plaintext = aes_decrypt(extracted_bytes, password)
                except Exception as e:
                    flash("Decryption failed: incorrect password or corrupted data.", "danger")
                    try: os.remove(img_path)
                    except: pass
                    return redirect(url_for("extract"))
            else:
                plaintext = extracted_bytes

            text = plaintext.decode("utf-8", errors="replace")
            flash("Extraction successful.", "success")
            try: os.remove(img_path)
            except: pass
            return render_template("extract.html", extracted_text=text)
        except Exception as e:
            flash(f"Error during extraction: {e}", "danger")
            try: os.remove(img_path)
            except: pass
            return redirect(url_for("extract"))

    return render_template("extract.html", extracted_text=None)

@app.route("/download/<filename>")
def download_file(filename):
    # serve from static/outputs
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    # use host=0.0.0.0 for LAN testing if needed
    app.run(debug=True)
