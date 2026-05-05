import os
import shutil
from html import escape
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from tachyphymia_model import load_tachyphymia_model, predict_tachyphemia
from emotion_model import load_emotion_model, predict_emotion
from fusion_rules import determine_bipolar_phase
from xai_audio import generate_emotion_xai


app = FastAPI(title="Bipolar Phase Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True)


tachy_model = load_tachyphymia_model("models/tachyphemia_cnn_final.keras")
emotion_processor, emotion_model, emotion_device = load_emotion_model("models/wav2vec2-emotion_final")

app.mount("/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")


def _run_analysis(file_path: Path, safe_filename: str, include_xai: bool = True) -> dict:
        tachy_result = predict_tachyphemia(str(file_path), tachy_model)

        emotion_result = predict_emotion(
                str(file_path),
                emotion_processor,
                emotion_model,
                emotion_device
        )

        phase_result = determine_bipolar_phase(
                tachy=tachy_result,
                emo=emotion_result
        )

        xai_result = None
        if include_xai and "error" not in emotion_result:
                try:
                        xai_result = generate_emotion_xai(
                                file_path=str(file_path),
                                original_name=safe_filename,
                                processor=emotion_processor,
                                model=emotion_model,
                                device=emotion_device,
                        )
                except Exception as xai_error:
                        xai_result = {
                                "error": f"XAI generation failed: {str(xai_error)}"
                        }

        return {
                "filename": safe_filename,
                "tachyphemia": tachy_result,
                "emotion": emotion_result,
                "phase_estimation": phase_result,
                "xai": xai_result,
                "ethical_warning": "This result is for monitoring only and is not a medical diagnosis."
        }


def _render_html_report(result: dict) -> str:
        filename = escape(str(result.get("filename", "unknown.wav")))

        tachy = result.get("tachyphemia", {})
        emotion = result.get("emotion", {})
        phase = result.get("phase_estimation", {})
        xai = result.get("xai") or {}

        waveform_url = escape(str(xai.get("plot_urls", {}).get("waveform", "")))
        spectrogram_url = escape(str(xai.get("plot_urls", {}).get("spectrogram", "")))

        important_segments = xai.get("important_segments_top5", [])
        segment_rows = ""
        for seg in important_segments:
                segment_rows += (
                        "<tr>"
                        f"<td>{escape(str(seg.get('start_sec', '')))}</td>"
                        f"<td>{escape(str(seg.get('end_sec', '')))}</td>"
                        f"<td>{escape(str(seg.get('mean_importance', '')))}</td>"
                        "</tr>"
                )

        if not segment_rows:
                segment_rows = "<tr><td colspan='3'>No important segments available.</td></tr>"

        xai_error_html = ""
        if "error" in xai:
                xai_error_html = f"<p class='error'>{escape(str(xai['error']))}</p>"

        waveform_html = ""
        if waveform_url:
                waveform_html = (
                        f"<h3>Waveform XAI</h3><img src='{waveform_url}' alt='Waveform XAI plot' />"
                )

        spectrogram_html = ""
        if spectrogram_url:
                spectrogram_html = (
                        f"<h3>Spectrogram XAI</h3><img src='{spectrogram_url}' alt='Spectrogram XAI plot' />"
                )

        return f"""
<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Bipolar Phase Monitor - XAI Report</title>
    <style>
        body {{
            font-family: "Segoe UI", Tahoma, sans-serif;
            background: linear-gradient(160deg, #f6f7fb 0%, #edf3ff 100%);
            color: #18212d;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1080px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(19, 39, 70, 0.12);
            padding: 24px;
        }}
        h1, h2, h3 {{ margin-top: 0; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .card {{
            border: 1px solid #e4eaf4;
            border-radius: 10px;
            padding: 12px 14px;
            background: #fbfdff;
        }}
        .k {{ color: #506178; font-size: 13px; margin-bottom: 6px; }}
        .v {{ font-size: 19px; font-weight: 600; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid #e4eaf4;
            padding: 8px 10px;
            text-align: left;
        }}
        th {{ background: #f4f7fc; }}
        img {{
            width: 100%;
            border: 1px solid #e4eaf4;
            border-radius: 10px;
            margin-bottom: 16px;
        }}
        .error {{
            color: #b42318;
            background: #ffe8e8;
            border: 1px solid #ffc9c9;
            padding: 10px;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class='container'>
        <h1>Bipolar Phase Monitor - Explainability Report</h1>
        <p><strong>Audio file:</strong> {filename}</p>

        <div class='grid'>
            <div class='card'>
                <div class='k'>Predicted Phase</div>
                <div class='v'>{escape(str(phase.get("phase", "N/A")))}</div>
            </div>
            <div class='card'>
                <div class='k'>Emotion</div>
                <div class='v'>{escape(str(emotion.get("emotion", "N/A")))}</div>
            </div>
            <div class='card'>
                <div class='k'>Tachyphemia Label</div>
                <div class='v'>{escape(str(tachy.get("cnn_label", "N/A")))}</div>
            </div>
            <div class='card'>
                <div class='k'>Agitation Score</div>
                <div class='v'>{escape(str(tachy.get("agitation_score", "N/A")))}</div>
            </div>
            <div class='card'>
                <div class='k'>Speech Rate (syll/s)</div>
                <div class='v'>{escape(str(tachy.get("speech_rate_syl_per_sec", "N/A")))}</div>
            </div>
            <div class='card'>
                <div class='k'>Emotion Confidence</div>
                <div class='v'>{escape(str(emotion.get("confidence", "N/A")))}</div>
            </div>
        </div>

        <h2>Clinical Explanation</h2>
        <p>{escape(str(phase.get("explanation", "No explanation available.")))}</p>

        <h2>Important Time Segments (Top 5)</h2>
        <table>
            <thead>
                <tr>
                    <th>Start (s)</th>
                    <th>End (s)</th>
                    <th>Mean Importance</th>
                </tr>
            </thead>
            <tbody>
                {segment_rows}
            </tbody>
        </table>

        <h2>XAI Visuals</h2>
        {xai_error_html}
        {waveform_html}
        {spectrogram_html}
    </div>
</body>
</html>
"""


@app.get("/")
def home():
    return {
        "message": "Bipolar Phase Monitor API is running",
        "input": "WAV audio file",
        "modules": [
            "CNN tachyphemia detection",
            "Wav2Vec2 emotion recognition",
            "clinical fusion rules"
        ],
        "warning": "Monitoring tool only. Not a medical diagnosis."
    }


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...), include_xai: bool = True):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav audio files are supported."
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return _run_analysis(file_path=file_path, safe_filename=safe_filename, include_xai=include_xai)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:
        if file_path.exists():
            file_path.unlink()


@app.post("/analyze-audio-report", response_class=HTMLResponse)
async def analyze_audio_report(file: UploadFile = File(...), include_xai: bool = True):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav audio files are supported."
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = _run_analysis(file_path=file_path, safe_filename=safe_filename, include_xai=include_xai)
        return HTMLResponse(content=_render_html_report(result), status_code=200)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )

    finally:
        if file_path.exists():
            file_path.unlink()