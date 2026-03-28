"""
Herald — Streamlit demo frontend
Runs on your Mac. Points to the Herald backend (FastAPI) on the A40 server.

Run:
    uv pip install -e ".[demo]"
    WHISPER_BACKEND_URL=http://<a40-host>:8000 streamlit run demo/app.py

Or set the URL in the sidebar at runtime.
"""

import os
import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Herald",
    page_icon="🎙️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar — backend config + health
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Config")
    backend_url = st.text_input(
        "Backend URL",
        value=os.getenv("WHISPER_BACKEND_URL", "http://localhost:8000"),
    ).rstrip("/")

    model = st.selectbox(
        "Model",
        options=["large-v3-turbo", "large-v3", "distil-large-v3"],
        index=0,
    )

    st.divider()
    st.subheader("Backend Health")

    try:
        r = requests.get(f"{backend_url}/health", timeout=3)
        r.raise_for_status()
        h = r.json()
        if h.get("model_loaded"):
            st.success("Connected")
        else:
            st.warning("Connected — model loading")
        st.metric("Model", h.get("model", "—"))
        st.metric("Backend", h.get("backend", "—").upper())
        if h.get("device_index") is not None:
            st.caption(f"GPU index: {h['device_index']}")
    except Exception as e:
        st.error(f"Backend offline\n\n`{e}`")

    st.divider()
    st.caption("On-prem inference. No audio leaves the network.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🎙️ Herald")
st.caption("Real-time speech transcription — A40 GPU inference")

tab_record, tab_upload = st.tabs(["🎤 Record", "📁 Upload"])


def display_result(result: dict):
    """Render transcription result with metrics."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Audio", f"{result['duration']}s")
    col2.metric("Processing", f"{result['processing_time']}s")
    col3.metric("RTF", result["rtf"])

    st.divider()

    lang = result.get("language", "?").upper()
    st.caption(f"Language: {lang}")

    st.subheader("Transcript")
    st.write(result["text"])

    with st.expander("Segments"):
        for seg in result.get("segments", []):
            st.markdown(
                f"`{seg['start']}s → {seg['end']}s` &nbsp; {seg['text']}",
                unsafe_allow_html=True,
            )


def run_transcription(audio_bytes: bytes, filename: str, mime: str):
    """POST audio to Herald backend and display result."""
    with st.spinner("Transcribing..."):
        try:
            r = requests.post(
                f"{backend_url}/v1/audio/transcriptions",
                data={"model": model, "response_format": "verbose_json"},
                files={"file": (filename, audio_bytes, mime)},
                timeout=120,
            )
            r.raise_for_status()
            display_result(r.json())
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend. Check the URL in the sidebar.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Backend error: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tab 1 — Record from mic
# ---------------------------------------------------------------------------
with tab_record:
    st.write("Record audio directly from your microphone.")
    audio = st.audio_input("Click to record")

    if audio:
        if st.button("Transcribe", type="primary", key="btn_record"):
            run_transcription(audio.getvalue(), "recording.wav", "audio/wav")

# ---------------------------------------------------------------------------
# Tab 2 — Upload a file
# ---------------------------------------------------------------------------
with tab_upload:
    st.write("Upload an existing audio file.")
    uploaded = st.file_uploader(
        "Audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
    )

    if uploaded:
        st.audio(uploaded)
        if st.button("Transcribe", type="primary", key="btn_upload"):
            run_transcription(uploaded.getvalue(), uploaded.name, uploaded.type)
