from flask import Flask, render_template, Response, jsonify, request, url_for
import os
import snapchat_engine
import cv2
import numpy as np
import time
import logging

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

@app.route("/save_snap", methods=["POST"])
def save_snap():
    try:
        current_filter = snapchat_engine.current_filter.lower()
        problematic_filters = ['cartoon', 'old film']
        # Retry up to 3 times to get a valid frame
        for attempt in range(3):
            chunk = next(snapchat_engine.gen_frames())
            try:
                # Log frame metadata for debugging
                logger.debug(f"Attempt {attempt + 1}: Frame size={len(chunk)} bytes, filter={current_filter}")
                logger.debug(f"First 10 bytes: {chunk[:10]}")
                
                # Extract image bytes
                img_bytes = chunk.split(b'\r\n\r\n')[1].split(b'\r\n')[0]
                arr = np.frombuffer(img_bytes, np.uint8)
                
                # Try decoding with different flags
                for flag in [cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYCOLOR]:
                    img = cv2.imdecode(arr, flag)
                    if img is not None and img.size > 0:
                        break
                
                if img is None or img.size == 0:
                    logger.warning(f"Invalid frame in attempt {attempt + 1} with flag={flag}, retrying...")
                    continue
                
                # Handle filter-specific issues
                if current_filter in problematic_filters:
                    # Convert grayscale to BGR (Old Film)
                    if len(img.shape) == 2 or img.shape[2] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # Handle non-standard channels (Cartoon)
                    elif img.shape[2] != 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
                
                # Save the image
                timestamp = int(time.time() * 1000)
                fn = f"snap_{timestamp}.jpg"
                filepath = os.path.join(snapchat_engine.PIC_DIR, fn)
                success = cv2.imwrite(filepath, img)
                if not success:
                    logger.warning(f"Failed to save image in attempt {attempt + 1}, retrying...")
                    continue
                
                url = url_for("static", filename=f"Pictures/{fn}")
                logger.debug(f"Snap saved: {filepath} with filter: {current_filter}")
                return jsonify({"url": url})
            
            except IndexError:
                logger.warning(f"Invalid frame format in attempt {attempt + 1}, retrying...")
                continue
            except Exception as e:
                logger.warning(f"Error processing frame in attempt {attempt + 1}: {e}, retrying...")
                continue
        
        # Fallback: Try saving a raw frame for problematic filters
        if current_filter in problematic_filters:
            logger.info(f"Fallback: Attempting to save raw frame for filter: {current_filter}")
            try:
                # Temporarily disable filter
                original_filter = snapchat_engine.current_filter
                snapchat_engine.set_filter('none')  # Assume 'none' is a valid filter
                chunk = next(snapchat_engine.gen_frames())
                snapchat_engine.set_filter(original_filter)  # Restore filter
                
                img_bytes = chunk.split(b'\r\n\r\n')[1].split(b'\r\n')[0]
                arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                
                if img is not None and img.size > 0:
                    timestamp = int(time.time() * 1000)
                    fn = f"snap_{timestamp}_raw.jpg"
                    filepath = os.path.join(snapchat_engine.PIC_DIR, fn)
                    cv2.imwrite(filepath, img)
                    url = url_for("static", filename=f"Pictures/{fn}")
                    logger.debug(f"Raw snap saved: {filepath}")
                    return jsonify({"url": url})
            except Exception as e:
                logger.error(f"Fallback failed for filter {current_filter}: {e}")
        
        logger.error(f"Failed to capture valid frame after 3 retries with filter: {current_filter}")
        return jsonify({"error": "Failed to save snap"}), 500
    
    except Exception as e:
        logger.error(f"Error saving snap with filter {current_filter}: {e}")
        return jsonify({"error": "Failed to save snap"}), 500

@app.route("/list_snaps")
def list_snaps():
    try:
        files = os.listdir(snapchat_engine.PIC_DIR)
        # Sort files by creation time (newest first)
        files.sort(key=lambda x: os.path.getctime(os.path.join(snapchat_engine.PIC_DIR, x)), reverse=True)
        urls = [url_for("static", filename=f"Pictures/{fn}") for fn in files]
        return jsonify({"snaps": urls})
    except Exception as e:
        logger.error(f"Error listing snaps: {e}")
        return jsonify({"error": "Failed to list snaps"}), 500

if __name__ == "__main__":
    app.run(debug=True)