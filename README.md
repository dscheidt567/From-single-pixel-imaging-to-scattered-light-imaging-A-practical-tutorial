# From Single-Pixel Imaging to Scattered Light Imaging — A Practical Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dscheidt567/From-single-pixel-imaging-to-scattered-light-imaging-A-practical-tutorial/blob/main/Tutorial_CS_SLI.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23825920.svg)](https://doi.org/10.5281/zenodo.23825920)

This repository contains the Jupyter notebook accompanying the journal article
**“From single-pixel imaging to scattered light imaging: a practical tutorial”**.

Scattered light imaging (SLI) determines nerve fiber orientations from the angular scattering profiles of histological sections. The tutorial applies single-pixel imaging (SPI) and compressive sensing (CS) to SLI: the tissue is illuminated with binary Hadamard patterns over the illumination angle, and every camera pixel acts as a bucket detector for that angle. All steps are applied to measured data of two crossing human optic tract sections. The notebook reproduces the analyses, figures and tables of the article and runs on the free Google Colab CPU runtime.

---

## Contents

| File | Description |
|---|---|
| `Tutorial_CS_SLI.ipynb` | The tutorial notebook (main entry point) |
| `sli_helpers.py` | Generic helpers: reading frames, ROIs, metrics, plotting, saving results |
| `sli_data.py` | Downloads the data on first use and provides the quick run |

The notebook is organised as follows:

0. Setup, data loading, overview of the section and choice of the regions of interest (ROIs)
1. Full-basis sampling: canonical vs. Hadamard SLI and the offset correction of the Hadamard data
2. The conditions of CS for SLI profiles: sparsity (DCT) and incoherence
3. Influence of the Hadamard ordering (natural, Walsh, cake-cutting, …)
4. Comparison of reconstruction algorithms (BP, BPDN, LASSO, ISTA, least squares)
5. Noise robustness: canonical vs. Hadamard vs. random patterns under shot and read noise
6. Learned linear reconstruction trained on measured profiles
7. Fiber orientation maps from compressed data and runtime extrapolation to full resolution
8. Summary and exercises

---

## Recommended Usage

Click the **Open in Colab** badge above and select *Runtime → Run all*. Packages and data are installed and downloaded automatically.

The notebook offers two modes, set by `QUICK_RUN` in Section 0.1:

- **Quick run** (default): uses a small recording of all data the published analysis reads from the raw frames (43 × 43 binned stacks and the four 100 × 100 px ROIs). Settings, figures and tables are the same as in the article. Changing the ROIs or the binning requires the full data.
- **Full run** (`QUICK_RUN = False`): uses the raw TIFF frames (canonical and Hadamard, N = 16, 32, 64; about 6 GB). The complete run takes about 20–25 min on the free Colab CPU runtime.

Results are written to `results_quick/` or `results/`. In Colab these folders are deleted with the runtime; the last cell of the notebook downloads them as a zip file. After switching between quick and full run, restart the Colab session (*Runtime → Restart session*).

To run the notebook locally:

```bash
git clone https://github.com/dscheidt567/From-single-pixel-imaging-to-scattered-light-imaging-A-practical-tutorial.git
cd From-single-pixel-imaging-to-scattered-light-imaging-A-practical-tutorial
pip install numpy scipy pandas matplotlib spgl1 tifffile torch gdown jupyter
jupyter notebook Tutorial_CS_SLI.ipynb
```

---

## Data

The data are published on Zenodo under CC BY 4.0: [doi:10.5281/zenodo.23825920](https://doi.org/10.5281/zenodo.23825920). They are downloaded automatically by `sli_data.py`; nothing has to be downloaded by hand. If you already have a copy, set the environment variable `SLI_DATA_ROOT` (raw frames) or `SLI_QUICK_ROOT` (quick data) to its location.

The measurements were acquired with a programmable LED panel and a 5472 × 3648 px camera (about 1.5 µm per pixel) and are described in Scheidt et al., *Biomedical Optics Express* (2026).

---

## Related Publications

- **From single-pixel imaging to scattered light imaging: a practical tutorial**
  *(link to be added once published)*
- D. Scheidt, F. Matuschke, K. Amunts, M. Menzel, M. Axer. “Hadamard-encoded scattered light imaging for faster, signal-enhanced mapping of brain fiber orientations.” *Biomedical Optics Express* 17(7), 3854–3876 (2026). [doi:10.1364/BOE.596576](https://doi.org/10.1364/BOE.596576)
- D. Scheidt et al. “Fast brain fiber mapping with compressive Hadamard-encoded scattered light imaging.” *Proc. SPIE* 14229, 142291P (2026). [doi:10.1117/12.3110655](https://doi.org/10.1117/12.3110655)
- Previous tutorial: [Single-Pixel Imaging and Compressive Sensing — A Practical Tutorial](https://github.com/dscheidt567/Single-Pixel-Imaging-and-Compressive-Sensing---A-practical-Tutorial)

---

## Citation

If you use this code or the data in your work, please cite the article (citation will be added once available) and the dataset:

> D. Scheidt. Dataset for “From single-pixel imaging to scattered light imaging: a practical tutorial”. Zenodo (2026). doi:10.5281/zenodo.23825920 GitHub’s *Cite this repository* button provides the citation of the software release.

---

## Acknowledgments

Partially funded by AIDAS (Joint Virtual Laboratory of Forschungszentrum Jülich and CEA) and by the European Union’s Horizon Europe Programme under Specific Grant Agreement No. 101147319 (EBRAINS 2.0).

If you have questions or suggestions, feel free to open an issue or get in touch.
