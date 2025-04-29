from flask import Flask, render_template, Response, jsonify, request, url_for, make_response
import os
import snapchat_engine
import cv2
import numpy as np
import time
import logging
import base64
import re

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_user_dir(username):
    """Get user-specific directory for snaps, fallback to PIC_DIR if invalid."""
    if not username or not re.match(r'^[a-zA-Z0-9_-]+$', username):
        logger.warning(f"Invalid or missing username: {username}, using default directory {snapchat_engine.PIC_DIR}")
        return snapchat_engine.PIC_DIR
    user_dir = os.path.join(snapchat_engine.PIC_DIR, username)
    try:
        os.makedirs(user_dir, exist_ok=True)
        if not os.access(user_dir, os.W_OK):
            logger.error(f"No write permission for directory: {user_dir}")
            return snapchat_engine.PIC_DIR
        return user_dir
    except Exception as e:
        logger.error(f"Failed to create directory {user_dir}: {e}")
        return snapchat_engine.PIC_DIR

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signin")
def signin():
    return render_template("signin.html")

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
        return jsonify({"error": f"Failed to set filter: {str(e)}"}), 500

@app.route("/set_intensity", methods=["POST"])
def set_intensity():
    try:
        value = request.json.get("intensity")
        snapchat_engine.set_intensity(value)
        return ("", 204)
    except Exception as e:
        logger.error(f"Error setting intensity: {e}")
        return jsonify({"error": f"Failed to set intensity: {str(e)}"}), 500

@app.route("/save_snap", methods=["POST"])
def save_snap():
    try:
        username = request.cookies.get('username')
        user_dir = get_user_dir(username)
        overlay_data = None
        if request.json and "overlay" in request.json:
            overlay_b64 = request.json["overlay"].split(",")[1] if "," in request.json["overlay"] else request.json["overlay"]
            overlay_data = base64.b64decode(overlay_b64)
        fname = snapchat_engine.save_snapshot(overlay_data, save_dir=user_dir)
        if fname is None:
            logger.error(f"Failed to capture snap with filter: {snapchat_engine.current_filter}")
            return jsonify({"error": "Failed to save snap: capture returned None"}), 500
        snap_path = os.path.join(user_dir, fname)
        if not os.path.exists(snap_path):
            logger.error(f"Snap file not found after saving: {snap_path}")
            return jsonify({"error": "Failed to save snap: file not created"}), 500
        url = url_for("static", filename=f"Pictures/{username}/{fname}" if username and user_dir != snapchat_engine.PIC_DIR else f"Pictures/{fname}")
        logger.debug(f"Snap saved: {snap_path}, URL: {url}")
        return jsonify({"url": url, "type": "image"})
    except Exception as e:
        logger.error(f"Error saving snap with filter {snapchat_engine.current_filter}: {e}")
        return jsonify({"error": f"Failed to save snap: {str(e)}"}), 500

@app.route("/record_video", methods=["POST"])
def record_video():
    try:
        username = request.cookies.get('username')
        user_dir = get_user_dir(username)
        duration = request.json.get("duration", 5)
        fname = snapchat_engine.record_video(duration, save_dir=user_dir)
        if fname is None:
            logger.error(f"Failed to record video with filter: {snapchat_engine.current_filter}")
            return jsonify({"error": "Failed to record video: capture returned None"}), 500
        video_path = os.path.join(user_dir, fname)
        if not os.path.exists(video_path):
            logger.error(f"Video file not found after recording: {video_path}")
            return jsonify({"error": "Failed to record video: file not created"}), 500
        url = url_for("static", filename=f"Pictures/{username}/{fname}" if username and user_dir != snapchat_engine.PIC_DIR else f"Pictures/{fname}")
        logger.debug(f"Video saved: {video_path}, URL: {url}")
        return jsonify({"url": url, "type": "video"})
    except Exception as e:
        logger.error(f"Error recording video: {e}")
        return jsonify({"error": f"Failed to record video: {str(e)}"}), 500

@app.route("/record_boomerang", methods=["POST"])
def record_boomerang():
    try:
        username = request.cookies.get('username')
        user_dir = get_user_dir(username)
        fname = snapchat_engine.record_boomerang(save_dir=user_dir)
        if fname is None:
            logger.error(f"Failed to record boomerang with filter: {snapchat_engine.current_filter}")
            return jsonify({"error": "Failed to record boomerang: capture returned None"}), 500
        boomerang_path = os.path.join(user_dir, fname)
        if not os.path.exists(boomerang_path):
            logger.error(f"Boomerang file not found after recording: {boomerang_path}")
            return jsonify({"error": "Failed to record boomerang: file not created"}), 500
        url = url_for("static", filename=f"Pictures/{username}/{fname}" if username and user_dir != snapchat_engine.PIC_DIR else f"Pictures/{fname}")
        logger.debug(f"Boomerang saved: {boomerang_path}, URL: {url}")
        return jsonify({"url": url, "type": "video"})
    except Exception as e:
        logger.error(f"Error recording boomerang: {e}")
        return jsonify({"error": f"Failed to record boomerang: {str(e)}"}), 500

@app.route("/list_snaps")
def list_snaps():
    try:
        username = request.cookies.get('username')
        user_dir = get_user_dir(username)
        files = os.listdir(user_dir) if os.path.exists(user_dir) else []
        files.sort(key=lambda x: os.path.getctime(os.path.join(user_dir, x)), reverse=True)
        snaps = []
        for fn in files:
            url = url_for("static", filename=f"Pictures/{username}/{fn}" if username and user_dir != snapchat_engine.PIC_DIR else f"Pictures/{fn}")
            file_type = "video" if fn.endswith(".mp4") else "image"
            snaps.append({"url": url, "type": file_type})
        logger.debug(f"Listed {len(snaps)} snaps for user: {username}")
        return jsonify({"snaps": snaps})
    except Exception as e:
        logger.error(f"Error listing snaps for user {username}: {e}")
        return jsonify({"error": f"Failed to list snaps: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)