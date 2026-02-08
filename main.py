import os
import subprocess
import tempfile
import uuid

import numpy as np
import imageio_ffmpeg
import streamlit as st
import torch
from df.enhance import enhance, init_df
from scipy.signal import butter, lfilter
from scipy.io import wavfile

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_path)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def apply_pro_equalization(audio: np.ndarray, fs: int) -> np.ndarray:
    """Podcast-style EQ: high-pass cleanup, bass lift, treble lift, then normalize."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    nyq = 0.5 * fs

    b, a = butter(4, 80 / nyq, btype="high")
    samples = lfilter(b, a, samples)

    b, a = butter(2, 150 / nyq, btype="low")
    bass = lfilter(b, a, samples)
    samples = samples + (bass * 0.3)

    b, a = butter(2, 5000 / nyq, btype="high")
    treble = lfilter(b, a, samples)
    samples = samples + (treble * 0.2)

    peak = np.max(np.abs(samples)) or 1.0
    samples = samples / peak * 0.9
    return samples.astype(np.float32)


def convert_to_wav(uploaded_file, target_sr: int) -> tuple[str, str]:
    """Save upload to disk and convert to mono WAV via bundled ffmpeg."""
    temp_dir = tempfile.gettempdir()
    file_format = uploaded_file.name.split(".")[-1].lower()
    src_path = os.path.join(temp_dir, f"temp_source_{uuid.uuid4().hex}.{file_format}")
    with open(src_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    temp_input = os.path.join(temp_dir, f"temp_input_{uuid.uuid4().hex}.wav")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        src_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        temp_input,
    ]

    try:
        completed = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)
    except Exception:
        cleanup_temp([temp_input, src_path])
        raise

    return src_path, temp_input


def load_wav_tensor(path: str) -> tuple[torch.Tensor, int]:
    """Load a mono WAV to float32 tensor shaped [1, T]."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_int
    tensor = torch.from_numpy(data).unsqueeze(0)
    return tensor, sr


def save_wav_np(path: str, audio: np.ndarray, sr: int) -> None:
    """Save float32 audio [-1,1] to 16-bit PCM WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    int_audio = (audio * 32767.0).astype(np.int16)
    wavfile.write(path, sr, int_audio)


def cleanup_temp(paths: list[str]) -> None:
    """Remove temporary files best-effort."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


st.set_page_config(page_title="AI Audio Enhancer", page_icon="EQ")

st.title("AI Audio Enhancer")
st.markdown("Transforme audios com IA DeepFilterNet e equalizacao de podcast.")

uploaded_file = st.file_uploader(
    "Escolha um arquivo de audio (wav, mp3, m4a, mp4)",
    type=["wav", "mp3", "m4a", "mp4"],
)

if uploaded_file is not None:
    file_format = uploaded_file.name.split(".")[-1].lower()

    # Preview respects the uploaded mime type; fall back to audio preview when unknown.
    if uploaded_file.type == "video/mp4":
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file, format=uploaded_file.type or "audio/wav")

    if st.button("Melhorar audio"):
        with st.spinner("A IA esta reconstruindo sua voz..."):
            model, df_state, _ = init_df()
            sr_attr = getattr(df_state, "sr", None)
            if callable(sr_attr):
                target_sr = int(sr_attr())
            elif isinstance(sr_attr, (int, float)):
                target_sr = int(sr_attr)
            else:
                target_sr = 48_000

            temp_paths: list[str] = []
            try:
                temp_src, temp_input = convert_to_wav(uploaded_file, target_sr)
                temp_paths.extend([temp_input, temp_src])

                audio, _ = load_wav_tensor(temp_input)
                enhanced = enhance(model, df_state, audio)
                enhanced_np = enhanced.squeeze(0).detach().cpu().numpy()
                enhanced_np = apply_pro_equalization(enhanced_np, target_sr)

                output_path = "output_final.wav"
                save_wav_np(output_path, enhanced_np, target_sr)

                st.success("Pronto!")

                st.markdown("### Resultado final")
                st.audio(output_path)

                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Baixar audio masterizado",
                        data=file,
                        file_name="audio_masterizado.wav",
                        mime="audio/wav",
                    )
            except RuntimeError as err:
                st.error("Falha ao converter o arquivo com o ffmpeg incorporado.")
                st.text(err)
                st.stop()
            finally:
                cleanup_temp(temp_paths)
