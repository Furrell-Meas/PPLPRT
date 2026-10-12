"""
app.py
User Interface + Output subsystem for the Privacy-Preserving Licence Plate
Redaction Tool (PPLPRT). Wires together the routes described in the
functional decomposition diagram:

    Upload Image File -> Validate Image File -> Store Image Temporarily
    -> processMedia() -> Display Side-by-Side Comparison
    -> Download Redacted Image
"""

import os
import uuid
import cv2
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash
)
from werkzeug.utils import secure_filename

from redaction import MediaFile, ValidationError, redact_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "pplprt-dev-secret"  # replace before any real deployment
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB, matches validate()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Handles: Upload -> Validate -> processMedia -> Display Comparison."""
    if "image" not in request.files or request.files["image"].filename == "":
        flash("Please choose an image file to upload.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    job_id = uuid.uuid4().hex
    safe_name = secure_filename(file.filename)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext or 'bin'}")
    file.save(upload_path)

    # --- Validate Image File ---
    media = MediaFile(upload_path, safe_name)
    try:
        img = media.validate()
    except ValidationError as e:
        os.remove(upload_path)
        flash(str(e), "error")
        return redirect(url_for("index"))

    # --- processMedia(media): detect -> filter -> expand -> blur ---
    redacted_img, boxes, plates_found = redact_image(img)

    output_filename = f"{job_id}_redacted.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, redacted_img)

    original_filename = f"{job_id}_original.png"
    original_path = os.path.join(OUTPUT_DIR, original_filename)
    cv2.imwrite(original_path, img)

    if not plates_found:
        flash("No licence plates were detected in this image. "
              "The downloaded file is unchanged from the original.", "warning")

    return render_template(
        "result.html",
        original_file=original_filename,
        redacted_file=output_filename,
        plates_found=plates_found,
        num_plates=len(boxes),
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serves images for inline display (Display Side-by-Side Comparison)."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/download/<path:filename>")
def download(filename):
    """Download Redacted Image step."""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True,
                                download_name="redacted_" + filename)


@app.errorhandler(413)
def file_too_large(e):
    flash("File too large. Maximum allowed size is 10MB.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
