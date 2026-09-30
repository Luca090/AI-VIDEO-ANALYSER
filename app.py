from flask import Flask, render_template, request
import os
import uuid

from analyzer import VideoAnalyzer


app = Flask(__name__)

# =====================================================
# CONFIGURATION
# =====================================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    500 * 1024 * 1024
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =====================================================
# LOAD AI
# =====================================================

print("Starting AI system...")

analyzer = VideoAnalyzer()

print("AI system ready.")


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        result=None,
        error=None
    )


# =====================================================
# ANALYZE
# =====================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    if "video" not in request.files:

        return render_template(
            "index.html",
            result=None,
            error="No video selected."
        )

    video = request.files["video"]

    if video.filename == "":

        return render_template(
            "index.html",
            result=None,
            error="Please select a video."
        )

    # -------------------------------------------------
    # CREATE UNIQUE FILE NAME
    # -------------------------------------------------

    extension = os.path.splitext(
        video.filename
    )[1]

    filename = (
        str(uuid.uuid4())
        + extension
    )

    video_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    video.save(
        video_path
    )

    print()
    print("Video saved:")
    print(video_path)

    # -------------------------------------------------
    # ANALYZE
    # -------------------------------------------------

    result = analyzer.analyze_video(
        video_path
    )

    # -------------------------------------------------
    # ERROR
    # -------------------------------------------------

    if not result.get("success"):

        return render_template(
            "index.html",
            result=None,
            error=result.get(
                "error",
                "Video analysis failed."
            )
        )

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------

    return render_template(
        "index.html",
        result=result,
        error=None
    )


# =====================================================
# START FLASK
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )