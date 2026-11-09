import os
import base64
import tempfile
import urllib.parse
import cv2
import numpy as np
import gradio as gr
import faceflux as ff

# --- Load branding assets from assets/ (relative paths) ---
assets_dir = "assets"
font_path = os.path.join(assets_dir, "kindsynapse-sans-wide-regular.ttf")
logo_path = os.path.join(assets_dir, "brand_tm.png")

clean_logo_path = logo_path.replace("\\", "/")
logo_url_path = f"/gradio_api/file={urllib.parse.quote(clean_logo_path)}"

font_base64 = ""
try:
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            font_base64 = base64.b64encode(f.read()).decode("utf-8")
except Exception as e:
    print(f"Warning: Could not load custom font: {e}")


def process_video(
    video_path, ref_img, trans, res, step, feather_pct, 
    progress=gr.Progress()
):
    # --- Initialization ---
    step = 5 - step
    padding_pct = 0.24

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress(0.0, desc="Initializing")

    # --- Prepare masked reference ---
    ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
    mask_ref, ref_box = ff.get_face_mask_box(ref_bgr, feather_pct, padding_pct)
    if mask_ref is None:
        progress(None)
        return None
    x_r, y_r, w_r, h_r = ref_box
    ref_cut = ref_bgr[y_r:y_r+h_r, x_r:x_r+w_r]
    mask_ref_norm = mask_ref.astype(np.float32)[..., None] / 255.0
    ref_masked = (ref_cut.astype(np.float32) * mask_ref_norm).astype(np.uint8)
    ref_morph = cv2.resize(ref_masked, (res, res))
    progress(0.1, desc="Reference ready")

    # --- Output setup ---
    w_o = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_o = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out_vid = cv2.VideoWriter(tmp_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_o, h_o))

    # --- Frame-by-frame processing ---
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        progress(0.1 + 0.8 * (i / total), desc=f"Processing frame {i+1}/{total}")

        mask_roi, box = ff.get_face_mask_box(frame, feather_pct, padding_pct)
        if mask_roi is None:
            out_vid.write(frame)
            continue
        x, y, w, h = box
        crop = frame[y:y+h, x:x+w]
        crop_resized = cv2.resize(crop, (res, res))
        alpha = float(np.clip((trans + 1) / 2, 0, 1))
        mor = ff.morph_faces(crop_resized, ref_morph, alpha, res, step)

        mor_back = cv2.resize(mor, (w, h))
        mask_n = (mask_roi.astype(np.float32)[..., None] / 255.0)
        region = frame[y:y+h, x:x+w].astype(np.float32)
        blended = region * (1 - mask_n) + mor_back.astype(np.float32) * mask_n
        frame[y:y+h, x:x+w] = blended.astype(np.uint8)
        out_vid.write(frame)

    cap.release()
    out_vid.release()

    progress(1.0, desc="Done")
    return tmp_vid


css = """
@font-face {
    font-family: 'KindSynapse';
    src: url('data:font/ttf;base64,FONT_BASE64_PLACEHOLDER') format('truetype');
}
video, img { object-fit: contain !important; }
.sulphur-point-title {
    font-family: 'KindSynapse', sans-serif !important;
    font-weight: 700;
    font-size: 2.55em;
    margin-bottom: 0.2em;
    letter-spacing: 0.05em;
}
.description-text {
    font-size: 1em;
    line-height: 1.6;
    margin-bottom: 1em;
}
.socials {
    font-size: 0.9em;
    color: #555;
    margin-bottom: 1.5em;
}
.socials li {
    margin-bottom: 0.2em;
}
.example-section {
    margin: 1em 0;
}
#brand-logo {
    max-height: 40px !important;
    width: auto !important;
    margin: 3em auto 1em auto !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#brand-logo img {
    max-height: 40px !important;
    width: auto !important;
    object-fit: contain !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
"""

css = css.replace("FONT_BASE64_PLACEHOLDER", font_base64)

with gr.Blocks(css=css) as iface:
    # Title
    gr.HTML('<div class="sulphur-point-title">FaceFlux</div>')

    # Description
    gr.Markdown(
        """
<div class="description-text">
<b><i>Super Fast Face Swap</i></b> using a CPU-friendly algorithm.
<br><br>
<i><b>How to Use:</b></i>
<ul>
  <li>Upload a video (ideal for small or mid-distance faces).</li>
  <li>Upload a clear reference image.</li>
  <li><b>Strength:</b> How much of the reference face to blend in.</li>
  <li><b>Quality:</b> Higher = better alignment, but slower.</li>
  <li><b>Feather:</b> Smooths edges for natural blending.</li>
  <li><b>Face Resolution:</b> Higher = sharper face but longer processing.</li>
</ul>
</div>
<br><br>
        """
    )

    # Socials
    gr.Markdown(
        """
<ul class="socials">
  <li>🌐 Website: <a href="https://nihalgazi.com" target="_blank">nihalgazi.com</a></li>
  <li>📷 Instagram: <a href="https://instagram.com/nihal_gazi_io" target="_blank">@nihal_gazi_io</a></li>
  <li>🐦 X (Twitter): <a href="https://x.com/NihalGazi_" target="_blank">@NihalGazi_</a></li>
  <li>💬 Discord: nihal_gazi_io</li>
  <li>📧 Email: info@nihalgazi.com</li>
</ul>
        """
    )

    # Inputs
    with gr.Row():
        vid = gr.Video(label='Input Video')
        ref = gr.Image(type='numpy', label='Reference Image')

    with gr.Row():
        res = gr.Dropdown([256, 384, 512, 768], value=512, label='Face Resolution')
        quality = gr.Slider(1, 4, value=1, step=1, label='Quality')
        feather = gr.Slider(0.12, 0.24, value=0.12, step=0.01, label='Feather (%)')
        strength = gr.Slider(-0.70, -0.15, value=-0.35, step=0.05, label='Strength')

    btn = gr.Button('Generate Morph 🚀')
    out_vid = gr.Video(label='Morphed Video')

    # Example (after component definitions)
    gr.Examples(
        examples=[
            ["examples/input.mp4", "examples/reference.jpg", -0.35, 512, 1, 0.12]
        ],
        inputs=[vid, ref, strength, res, quality, feather],
        outputs=[out_vid],
        examples_per_page=1,
        cache_examples=False
    )

    btn.click(
        fn=process_video,
        inputs=[vid, ref, strength, res, quality, feather],
        outputs=[out_vid],
        show_progress=True,
        concurrency_limit=30
    )

    gr.HTML(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 3em; margin-bottom: 1em;">
            <img src="{logo_url_path}" style="max-height: 40px; width: auto; opacity: 0.85;" alt="KindSynapse Brand TM">
        </div>
        """
    )

if __name__ == "__main__":
    allowed_paths = [p for p in [font_path, logo_path] if os.path.exists(p)]
    iface.queue().launch(debug=True, allowed_paths=allowed_paths)
