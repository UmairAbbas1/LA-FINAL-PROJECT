from flask import Flask, render_template, Response, jsonify, request, url_for
import os
import snapchat_engine
import cv2
import numpy as np
import time
import logging
import base64

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        snapchat_engine.gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/filters")
def list_filters():
    return jsonify({
        "filters": snapchat_engine.filter_names,
        "current": snapchat_engine.current_filter
    })

@app.route("/set_filter/<name>", methods=["POST"])
def set_filter(name):
    try:
        snapchat_engine.set_filter(name)
        return ("", 204)
    except Exception as e:
        logger.error(f"Error setting filter {name}: {e}")
        return jsonify({"error": "Failed to set filter"}), 500

@app.route("/set_intensity", methods=["POST"])
def set_intensity():
    try:
        value = request.json.get("intensity")
        snapchat_engine.set_intensity(value)
        return ("", 204)
    except Exception as e:
        logger.error(f"Error setting intensity: {e}")
        return jsonify({"error": "Failed to set intensity"}), 500

@app.route("/save_snap", methods=["POST"])
def save_snap():
    try:
        overlay_data = None
        if request.json and "overlay" in request.json:
            overlay_b64 = request.json["overlay"].split(",")[1] if "," in request.json["overlay"] else request.json["overlay"]
            overlay_data = base64.b64decode(overlay_b64)
        fname = snapchat_engine.save_snapshot(overlay_data)
        if fname is None:
            logger.error(f"Failed to capture snap with filter: {snapchat_engine.current_filter}")
            return jsonify({"error": "Failed to save snap"}), 500
        url = url_for("static", filename=f"Pictures/{fname}")
        logger.debug(f"Snap saved: {os.path.join(snapchat_engine.PIC_DIR, fname)}")
        return jsonify({"url": url, "type": "image"})
    except Exception as e:
        logger.error(f"Error saving snap with filter {snapchat_engine.current_filter}: {e}")
        return jsonify({"error": "Failed to save snap"}), 500

@app.route("/record_video", methods=["POST"])
def record_video():
    try:
        duration = request.json.get("duration", 5)
        fname = snapchat_engine.record_video(duration)
        if fname is None:
            logger.error(f"Failed to record video with filter: {snapchat_engine.current_filter}")
            return jsonify({"error": "Failed to record video"}), 500
        url = url_for("static", filename=f"Pictures/{fname}")
        logger.debug(f"Video saved: {os.path.join(snapchat_engine.PIC_DIR, fname)}")
        return jsonify({"url": url, "type": "video"})
    except Exception as e:
        logger.error(f"Error recording video: {e}")
        return jsonify({"error": "Failed to record video"}), 500

@app.route("/record_boomerang", methods=["POST"])
def record_boomerang():
    try:
        fname = snapchat_engine.record_boomerang()
        if fname is None:
            logger.error(f"Failed to record boomerang with filter: {snapchat_engine.current_filter}")
            return jsonify({"error": "Failed to record boomerang"}), 500
        url = url_for("static", filename=f"Pictures/{fname}")
        logger.debug(f"Boomerang saved: {os.path.join(snapchat_engine.PIC_DIR, fname)}")
        return jsonify({"url": url, "type": "video"})
    except Exception as e:
        logger.error(f"Error recording boomerang: {e}")
        return jsonify({"error": "Failed to record boomerang"}), 500

@app.route("/list_snaps")
def list_snaps():
    try:
        files = os.listdir(snapchat_engine.PIC_DIR)
        files.sort(key=lambda x: os.path.getctime(os.path.join(snapchat_engine.PIC_DIR, x)), reverse=True)
        snaps = []
        for fn in files:
            url = url_for("static", filename=f"Pictures/{fn}")
            file_type = "video" if fn.endswith(".mp4") else "image"
            snaps.append({"url": url, "type": file_type})
        return jsonify({"snaps": snaps})
    except Exception as e:
        logger.error(f"Error listing snaps: {e}")
        return jsonify({"error": "Failed to list snaps"}), 500

if __name__ == "__main__":
    app.run(debug=True)