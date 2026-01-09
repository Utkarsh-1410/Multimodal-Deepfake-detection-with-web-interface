## Multimodal DeepFake Detection (MRI-GAN + Temporal + Web)

This repository is a practical DeepFake detection toolkit built around multiple complementary signals:

- **Plain frames** CNN baseline
- **MRI-based features** (perceptual-difference “MRI” maps from an MRI-GAN-style generator)
- **Fusion** of modalities
- **Temporal** scoring over a sliding window
- **Web UI (FastAPI)** for interactive video upload and inference

### What makes this repo different
This codebase is based on the MRI-GAN research idea (paper below) and expands it into a more end-to-end, runnable project with:

- A web interface and API under `web/`
- Batch inference scripts and enhanced pipelines
- Optional temperature calibration utilities
- Clearer “batteries included” project structure for experimentation

**Reference paper:** https://arxiv.org/abs/2203.00108

---

## MRI-GAN concept (high-level)

MRI-GAN generates an “MRI map” for an input face frame. For fake frames, the map tends to highlight synthesized regions; for real frames it tends to be near-black.

![MRI-GAN architecture](images/mri_model_arch.png)

### Training visuals

![Discriminator](images/dis_model.png)
![Generator](images/gen_model.png)

![MRI dataset formulation](images/mri_df_dataset_gen.png)

![MRI sample output](images/MRI_demo.png)

---

## Quickstart (inference)

### 1) Install

- If you use conda: `conda env create -f environment.yml`
- Or use pip: `pip install -r requirements.txt`

### 2) Configure

Edit `config.yml` (or `config_windows.yml` on Windows) to point to your local dataset / cache paths.

### 3) Run CLI inference

Use the predictor entrypoint in `deep_fake_detect_app.py` (the web server calls this too).

- Example:
  - `python deep_fake_detect_app.py --input_videofile <path> --method plain_frames`
  - `python deep_fake_detect_app.py --input_videofile <path> --method MRI`
  - `python deep_fake_detect_app.py --input_videofile <path> --method fusion`
  - `python deep_fake_detect_app.py --input_videofile <path> --method temporal`

---

## Web UI (FastAPI)

The web app lives under `web/` and provides a simple upload + prediction interface.

- Install web deps: `pip install -r web/requirements.txt`
- Run server: `uvicorn web.server:app --reload --port 8000`
- Open: http://127.0.0.1:8000

---

## Datasets & large files

Datasets, generated artifacts, and model weights are intentionally excluded from git (see `.gitignore`).
Use `download_models.py` or your own storage to manage weights.

---

## Citation / Credits

If you use MRI-GAN ideas academically, cite the original paper:

```
Pratikkumar Prajapati and Chris Pollett, MRI-GAN: A Generalized Approach to Detect DeepFakes using Perceptual Image Assessment. arXiv preprint arXiv:2203.00108 (2022)
```

This repository started from the public implementation at https://github.com/pratikpv/mri_gan_deepfake and was extended with additional pipelines and a web interface.
