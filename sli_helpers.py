"""
Helper functions for the tutorial notebook "Compressive Sensing for Scattered Light Imaging (SLI)".

This module contains everything that is *not* specific to single-pixel imaging or compressive sensing:
reading the raw TIFF frames, selecting regions of interest, generic metrics, figure styles, plotting and
saving of results. The physics and the reconstruction algorithms (Hadamard sampling, offset correction,
sparsity, solvers, learned reconstruction) are defined in the notebook itself.

Settings that several functions share (e.g. whether the first TIFF of a folder is skipped, the angle
convention, the result folders) are set once with `configure(...)` and `init_output(...)`.
"""
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import tifffile
from scipy import ndimage

# =============================================================================================== settings
SETTINGS = {
    "skip_first_frame": False,           # True / False / "auto" (skip only if a folder holds N+1 files)
    "angle_convention": "rotate_90_minus",
    "data_dirs": {},                     # manual override {(basis, n): folder}
    "sample_filter": "",
    "tile": "",
    "save_figures": True,
    "article_dir": None,
    "diag_dir": None,
}


def configure(**kwargs):
    """Set shared options, e.g. configure(skip_first_frame=False, angle_convention="as_is")."""
    unknown = set(kwargs) - set(SETTINGS)
    if unknown:
        raise KeyError(f"Unknown setting(s): {unknown}")
    SETTINGS.update(kwargs)
    _BIN_CACHE.clear()
    _ROI_CACHE.clear()


# =============================================================================================== output
def init_output(output_dir, save_figures=True):
    """Create results/article/{figures,tables} and results/diagnostics/{figures,tables}."""
    output_dir = Path(output_dir)
    article, diag = output_dir / "article", output_dir / "diagnostics"
    for d in (article, diag):
        (d / "figures").mkdir(parents=True, exist_ok=True)
        (d / "tables").mkdir(parents=True, exist_ok=True)
    SETTINGS.update(article_dir=article, diag_dir=diag, save_figures=save_figures)
    return article, diag


def savefig(fig, name, kind="diagnostic"):
    """Article figures: PDF + SVG (vector) and PNG at 600 dpi. Diagnostics: PNG at 200 dpi."""
    if not SETTINGS["save_figures"]:
        return
    if kind == "article":
        base = SETTINGS["article_dir"] / "figures" / name
        for ext in ("pdf", "svg"):
            fig.savefig(base.with_suffix("." + ext), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    else:
        fig.savefig((SETTINGS["diag_dir"] / "figures" / name).with_suffix(".png"), dpi=200, bbox_inches="tight")


def save_table(df, name, kind="diagnostic"):
    folder = SETTINGS["article_dir"] if kind == "article" else SETTINGS["diag_dir"]
    path = folder / "tables" / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def enable_cell_timer(min_seconds=2.0):
    """Print the run time below every notebook cell that takes longer than `min_seconds`."""
    try:
        ip = get_ipython()  # noqa: F821  (defined inside IPython / Jupyter / Colab)
    except NameError:
        return
    state = {}

    def pre(*_):
        state["t"] = time.perf_counter()

    def post(*_):
        dt = time.perf_counter() - state.get("t", time.perf_counter())
        if dt >= min_seconds:
            print(f"[cell run time: {dt / 60:.1f} min]" if dt >= 60 else f"[cell run time: {dt:.0f} s]")

    for event, fn in (("pre_run_cell", pre), ("post_run_cell", post)):
        for old in [f for f in ip.events.callbacks[event] if getattr(f, "_sli_timer", False)]:
            ip.events.unregister(event, old)
        fn._sli_timer = True
        ip.events.register(event, fn)


# =============================================================================================== styles
NOTEBOOK_RC = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "lines.linewidth": 2.0, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 200,
}

# Article figures are created at their printed size (IOP: single column 8.5 cm, double column 17.5 cm),
# so the fonts keep 7-9 pt in the journal and are not scaled down.
ART_W1, ART_W2 = 3.35, 6.9
ARTICLE_RC = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.frameon": False,
    "lines.linewidth": 1.2, "lines.markersize": 3, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 3,
    "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "svg.fonttype": "none",
}


def article_style():
    return plt.rc_context(ARTICLE_RC)


def panel_labels(axes, start=0):
    """Prefix (a), (b), ... to the panel titles; panels without a title get the label as left title."""
    for k, ax in enumerate(np.ravel(axes)):
        lab = f"({chr(97 + start + k)})"
        if ax.get_title():
            ax.set_title(f"{lab} {ax.get_title()}")
        elif ax.get_title(loc="left"):
            ax.set_title(f"{lab} {ax.get_title(loc='left')}", loc="left", fontweight="bold")
        else:
            ax.set_title(lab, loc="left", fontweight="bold")


COLORS = {"Natural Hadamard": "black", "Hadamard-Walsh": "#C00000", "Cake-cutting": "#0072B2",
          "Random binary": "#E69F00", "Canonical": "0.5"}
ALG_COLORS = {"Min-norm LS": "0.5", "ISTA": "#009E73", "SPGL1 BP": "#C00000", "SPGL1 BPDN": "#CC79A7",
              "SPGL1 LASSO": "#0072B2"}
ABBR = {"Natural Hadamard": "natural", "Hadamard-Walsh": "Walsh", "Cake-cutting": "cake"}
ROI_COLORS = ["#E69F00", "#D55E00", "#56B4E9", "#0072B2"]
CLASS_COLORS = ["#E69F00", "#56B4E9", "#7B3294", "#BBBBBB"]

# =============================================================================================== data
TIF_EXT = (".tif", ".tiff")


def natural_key(s):
    """Sort key so that img_2.tif comes before img_10.tif."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def list_tifs(directory):
    files = [f for f in os.listdir(directory) if f.lower().endswith(TIF_EXT)]
    return [os.path.join(directory, f) for f in sorted(files, key=natural_key)]


def frame_files(directory, n):
    """The n frames of one basis in acquisition order (see SETTINGS['skip_first_frame'])."""
    files = list_tifs(directory)
    skip = SETTINGS["skip_first_frame"]
    if skip == "auto":
        if len(files) not in (n, n + 1):
            raise ValueError(f"{directory} holds {len(files)} TIFFs, expected {n} or {n + 1}.")
        start = len(files) - n
    else:
        start = 1 if skip else 0
        if len(files) < start + n:
            raise ValueError(f"{directory} holds {len(files)} TIFFs, expected at least {start + n}.")
    return files[start:start + n]


def _matches(path, basis, n):
    tokens = {"can": ["can", "canonical"], "had": ["had", "hadamard"]}[basis]
    return any(re.search(rf"(^|[_\-\s]){tok}[_\-\s]?{n}($|[_\-\s])", part, flags=re.I)
               for part in Path(path).parts for tok in tokens)


def find_stack_dir(root, basis, n):
    """Folder below `root` whose name contains the basis and n (e.g. can_32), holding >= n TIFFs."""
    if (basis, n) in SETTINGS["data_dirs"]:
        return SETTINGS["data_dirs"][(basis, n)]
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        if SETTINGS["sample_filter"] and SETTINGS["sample_filter"].lower() not in dirpath.lower():
            continue
        if sum(f.lower().endswith(TIF_EXT) for f in filenames) >= n and _matches(os.path.relpath(dirpath, root), basis, n):
            candidates.append(dirpath)
    if SETTINGS["tile"]:
        candidates = [c for c in candidates if Path(c).name == SETTINGS["tile"]] or candidates
    return sorted(candidates, key=natural_key)[0] if candidates else None


def discover_stacks(root, ns):
    """{(basis, n): folder} for all N that have both a canonical and a Hadamard folder."""
    dirs = {}
    for n in ns:
        for basis in ("can", "had"):
            d = find_stack_dir(root, basis, n)
            if d is None:
                warnings.warn(f"No folder found for basis={basis}, N={n}.")
            else:
                dirs[(basis, n)] = d
                print(f"{basis}  N={n:3d}: {d}")
    complete = [n for n in ns if ("can", n) in dirs and ("had", n) in dirs]
    return dirs, complete


def read_frame(path):
    return tifffile.imread(path).astype(np.float64)


def bin_image(img, b):
    """Mean over b x b blocks (the image is cropped to a multiple of b)."""
    if b == 1:
        return img
    h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    return img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


_BIN_CACHE, _ROI_CACHE = {}, {}


def _roi_key(rois):
    return tuple(tuple(int(v) for v in r) for r in rois)


def load_stack_and_rois(directory, n, b, rois=()):
    """
    Read every frame of a stack once and return (binned stack (n, H/b, W/b), ROI profiles (n_rois, n)).
    Both results are cached: later calls of load_binned_stack / load_roi_profiles do not re-read the TIFFs.
    """
    key_b, key_r = (str(directory), int(n), int(b)), (str(directory), int(n), _roi_key(rois))
    need_b, need_r = key_b not in _BIN_CACHE, len(rois) > 0 and key_r not in _ROI_CACHE
    if need_b or need_r:
        stack, prof = [], np.zeros((len(rois), n))
        for k, f in enumerate(frame_files(directory, n)):
            img = read_frame(f)
            if need_b:
                stack.append(bin_image(img, b))
            for r, (x0, y0, w, h) in enumerate(rois):
                prof[r, k] = img[y0:y0 + h, x0:x0 + w].mean()
        if need_b:
            _BIN_CACHE[key_b] = np.stack(stack)
        if need_r:
            _ROI_CACHE[key_r] = prof
    return _BIN_CACHE[key_b].copy(), (_ROI_CACHE[key_r].copy() if len(rois) else None)


def load_binned_stack(directory, n, b):
    """Binned stack (n, H/b, W/b), cached."""
    return load_stack_and_rois(directory, n, b)[0]


def load_roi_profiles(directory, n, rois):
    """(n_rois, n): mean intensity of every ROI (x0, y0, w, h) in every frame, cached."""
    key = (str(directory), int(n), _roi_key(rois))
    if key not in _ROI_CACHE:
        prof = np.zeros((len(rois), n))
        for k, f in enumerate(frame_files(directory, n)):
            img = read_frame(f)
            for r, (x0, y0, w, h) in enumerate(rois):
                prof[r, k] = img[y0:y0 + h, x0:x0 + w].mean()
        _ROI_CACHE[key] = prof
    return _ROI_CACHE[key].copy()


# =============================================================================================== angles
def acquisition_angles(n):
    return np.linspace(0.0, 360.0, n, endpoint=False)


def physical_angles(n):
    """Physical illumination angle of every segment (SETTINGS['angle_convention'])."""
    a = acquisition_angles(n)
    if SETTINGS["angle_convention"] == "as_is":
        return a
    if SETTINGS["angle_convention"] == "rotate_90_minus":
        return (90.0 - a) % 360.0
    raise ValueError("angle_convention must be 'as_is' or 'rotate_90_minus'.")


def prepare_profile_for_plot(profile, n):
    """Sort a profile by physical angle and close it at 360 deg."""
    ang = physical_angles(n)
    order = np.argsort(ang)
    return np.append(ang[order], 360.0), np.append(profile[order], profile[order][0])


# =============================================================================================== metrics
def ncc(a, b):
    """Normalised cross-correlation (shape similarity, insensitive to offset and scale)."""
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    den = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return np.nan if den == 0 else float(np.sum(a * b) / den)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def ncc_rows(A, B):
    """NCC of every row of two (P, n) arrays; NaN for constant rows."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    den = np.sqrt(np.sum(A**2, 1) * np.sum(B**2, 1))
    return np.where(den > 0, np.sum(A * B, 1) / np.where(den > 0, den, 1), np.nan)


def nanmean_quiet(x):
    """Mean ignoring NaN, NaN (without warning) if all values are NaN."""
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def axial_difference(a, b):
    """Difference of two axes (period 180 deg), folded to [0, 90] deg."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)


def match_affine(z, ref, skip_first=True):
    """Least squares a*z + c ~ ref on segments 1..N-1; returns (a*z + c, a, c)."""
    idx = slice(1, None) if skip_first else slice(None)
    A = np.column_stack([z[idx], np.ones(z[idx].size)])
    (a, c), *_ = np.linalg.lstsq(A, ref[idx], rcond=None)
    return a * z + c, a, c


# =============================================================================================== regions and ROIs
def kmeans(X, k, restarts=10, n_iter=200, seed=0):
    """Plain k-means with k-means++ initialisation; returns (labels, centres) of the best restart."""
    rk = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(restarts):
        C = [X[rk.integers(len(X))]]
        for _ in range(k - 1):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in C], axis=0)
            C.append(X[rk.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C)
        for _ in range(n_iter):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            Cn = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inertia = np.sum((X - C[lab]) ** 2)
        if inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def classify_regions(axis_map, aniso_map, mask, seed=0, crossing_min_a2=0.3):
    """
    Split the section into strand A, strand B, crossing and isotropic pixels.

    Every pixel is a point p = A2 exp(2i theta). Single strands form clusters at 2*theta of the strand,
    crossing pixels lie between them, closer to the origin. Pixels close to the origin with a very small A2
    (isotropic, background) are labelled separately. Returns (class_map, class_names, class_phasors).
    """
    phasor = aniso_map * np.exp(2j * np.deg2rad(axis_map))
    X = np.column_stack([phasor[mask].real, phasor[mask].imag])
    lab, centers = kmeans(X, 3, seed=seed)
    c = centers[:, 0] + 1j * centers[:, 1]
    i_cross = int(np.argmin(np.abs(c)))
    i_strands = sorted([i for i in range(3) if i != i_cross], key=lambda i: (np.angle(c[i], deg=True) / 2) % 180)
    names = ["strand A", "strand B", "crossing", "isotropic / other"]
    class_map = np.full(mask.shape, -1)
    tmp = np.full(len(X), -1)
    for new, old in enumerate(i_strands + [i_cross]):
        tmp[lab == old] = new
    class_map[mask] = tmp
    phasors = {name: c[old] for name, old in zip(names, i_strands + [i_cross])}
    weak = (class_map == 2) & (aniso_map < crossing_min_a2 * min(abs(phasors["strand A"]), abs(phasors["strand B"])))
    class_map[weak] = 3
    return class_map, names, phasors


def auto_rois(class_map, img_shape, binning, roi_size=100, border_margin=200, exclude_rois=(), exclude_margin=300):
    """
    ROI 1 in strand A, ROI 2 in strand B, ROI 3 and 4 in the crossing: each ROI is placed at the point of its
    region with the largest distance to the region boundary, away from the image border and excluded regions.
    Returns a list of (x0, y0, w, h) in full-resolution pixels.
    """
    img_h, img_w = img_shape

    def allowed(mask):
        m = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mb = min(int(np.ceil(border_margin / binning)), min(m.shape) // 10)
        if mb > 0:
            m[:mb] = m[-mb:] = False
            m[:, :mb] = m[:, -mb:] = False
        e = int(np.ceil(exclude_margin / binning))
        for (x0, y0, w, h) in exclude_rois:
            ys, xs = y0 // binning, x0 // binning
            m[max(ys - e, 0):ys + h // binning + e + 1, max(xs - e, 0):xs + w // binning + e + 1] = False
        return m

    def place(mask, count, min_sep):
        a = allowed(mask)
        if not a.any():
            raise RuntimeError("No admissible pixel for an automatic ROI: reduce the margins or set ROIS by hand.")
        dist = ndimage.distance_transform_edt(a)
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        out = []
        for _ in range(count):
            d = dist.copy()
            for (cy, cx) in out:
                d[np.hypot(yy - cy, xx - cx) < min_sep] = 0
            if d.max() * binning < roi_size / 2:
                warnings.warn("A region is too small for the requested ROI size.")
            out.append(np.unravel_index(np.argmax(d), d.shape))
        return out

    def to_roi(yb, xb):
        cy, cx = int((yb + 0.5) * binning), int((xb + 0.5) * binning)
        x0 = int(np.clip(cx - roi_size // 2, 0, img_w - roi_size))
        y0 = int(np.clip(cy - roi_size // 2, 0, img_h - roi_size))
        return (x0, y0, roi_size, roi_size)

    sep = max(class_map.shape) // 4
    centres = place(class_map == 0, 1, sep) + place(class_map == 1, 1, sep) + place(class_map == 2, 2, sep)
    return [to_roi(*c) for c in centres]


# =============================================================================================== plotting
def plot_overview(mean_img, aniso_map, mask, class_map, class_names, rois, img_shape, calib_roi, roi_label):
    """Article figure (image, anisotropy, regions with ROIs) and a diagnostic version with coordinates."""
    img_h, img_w = img_shape
    ext = [0, img_w, img_h, 0]
    with article_style():
        fig, ax = plt.subplots(1, 3, figsize=(ART_W2, 1.95), constrained_layout=True)
        ax[0].imshow(mean_img, cmap="gray", extent=ext)
        ax[0].set_title("Mean canonical image")
        im1 = ax[1].imshow(np.where(mask, aniso_map, np.nan), cmap="magma", extent=ext)
        ax[1].set_title("Anisotropy $A_2$")
        cb = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.02, shrink=0.8)
        cb.ax.tick_params(labelsize=7)
        ax[2].imshow(np.ma.masked_less(class_map, 0), cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=3.5,
                     extent=ext, interpolation="nearest")
        ax[2].set_title("Regions")
        ax[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS[:3]],
                     labels=class_names[:3], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
                     handlelength=1.0, borderaxespad=0)
        for a in ax:
            for r, (x0, y0, w, h) in enumerate(rois):
                a.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="cyan", lw=0.8))
                a.plot(x0 + w / 2, y0 + h / 2, "o", ms=6, mfc="none", mec="cyan", mew=1.0)
                a.annotate(str(r + 1), (x0 + w / 2, y0 + h / 2), xytext=(4, 4), textcoords="offset points",
                           color="white", fontsize=7, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="cyan", lw=0.5, alpha=0.75))
            a.set_xticks([]); a.set_yticks([])
        panel_labels(ax)
        savefig(fig, "fig01_overview_rois", kind="article")
        plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].imshow(mean_img, cmap="gray", extent=ext); ax[0].set_title("Mean canonical image")
    im = ax[1].imshow(np.where(mask, aniso_map, np.nan), cmap="magma", extent=ext)
    ax[1].set_title("Anisotropy $A_2$"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        for r, (x0, y0, w, h) in enumerate(rois):
            a.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="cyan", lw=2))
            a.annotate(roi_label(r), (x0 + w, y0), xytext=(6, 6), textcoords="offset points", color="cyan", fontsize=10)
        x0, y0, w, h = calib_roi
        a.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="yellow", ls="--", lw=1.5))
        a.set_xlabel("x (px)"); a.set_ylabel("y (px)")
    fig.tight_layout(); savefig(fig, "diag00_overview_coordinates"); plt.show()


def plot_roi_profiles(profiles, axes_deg, a2, n, roi_label):
    with article_style():
        fig, axes = plt.subplots(1, len(profiles), figsize=(ART_W2, 1.75), sharey=True, squeeze=False,
                                 constrained_layout=True)
        for r, ax in enumerate(axes[0]):
            ax.plot(*prepare_profile_for_plot(profiles[r] / profiles[r].max(), n), "o-", ms=2, lw=1.0, color="0.2")
            for t in (axes_deg[r], (axes_deg[r] + 180) % 360):
                ax.axvline(t, color="#C00000", ls="--", lw=0.8)
            ax.set_title(f"{roi_label(r)}\n$A_2$ = {a2[r]:.2f}")
            ax.set_xticks(range(0, 361, 90)); ax.set_xlabel("Angle $\\varphi$ (deg)")
        axes[0, 0].set_ylabel("Norm. intensity")
        panel_labels(axes[0])
        savefig(fig, "fig02_roi_profiles", kind="article")
        plt.show()


def plot_calibration_ratio(x_raw, can, ns, roi_label):
    """Diagnostic: per-segment ratio raw Hadamard / canonical, normalised by its median."""
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 3.8), squeeze=False)
    for ax, n in zip(axes[0], ns):
        ratio = x_raw[n] / can[n]
        ratio /= np.median(ratio, axis=1, keepdims=True)
        for r in range(ratio.shape[0]):
            ax.plot(np.arange(n), ratio[r], "o-", ms=4, label=roi_label(r))
        ax.set_yscale("log"); ax.set_xlabel("Segment index j"); ax.set_title(f"N = {n}")
    axes[0, 0].set_ylabel("raw Hadamard / canonical"); axes[0, 0].legend(fontsize=10)
    fig.tight_layout(); savefig(fig, "diag02_calibration_ratio"); plt.show()


def plot_full_basis(data, ns, r_show, correction_label, show_corrected_coeffs):
    """Article figure: raw frames, signed coefficients and profiles for one ROI and every N."""
    C_CAN, C_HAD, C_RAW = "0.35", "#C00000", "#009E73"
    with article_style():
        fig, axes = plt.subplots(len(ns), 3, figsize=(ART_W2, 1.75 * len(ns) + 0.45), squeeze=False,
                                 constrained_layout=True)
        for i, n in enumerate(ns):
            d = data[(r_show, n)]
            k = np.arange(n)
            ax = axes[i, 0]; ax2 = ax.twinx()
            ax.plot(k, d["can_frames"], "o-", ms=2, color=C_CAN)
            ax2.plot(k, d["frames"], "s-", ms=2, color=C_RAW)
            ax.set_ylabel("Canonical (counts)", color=C_CAN); ax2.set_ylabel("H$^+$ (counts)", color=C_RAW)
            ax2.spines["right"].set_visible(True)
            ax2.tick_params(axis="y", colors=C_RAW); ax.tick_params(axis="y", colors=C_CAN)
            ax.set_title(f"N = {n}", loc="left", fontweight="bold")
            ax = axes[i, 1]
            ax.vlines(k[1:], 0, d["y_raw"][1:], color="C0", lw=0.8); ax.plot(k[1:], d["y_raw"][1:], "o", ms=2, color="C0")
            ax.vlines([0], 0, d["y_raw"][:1], color="C3", lw=0.8); ax.plot([0], d["y_raw"][:1], "o", ms=2.5, color="C3")
            if show_corrected_coeffs:
                ax.plot(k, d["y"], "_", color="k", ms=4, mew=0.9)
            ax.set_ylabel("$y_k$ (norm.)")
            ax = axes[i, 2]
            ax.plot(*prepare_profile_for_plot(d["z_can_matched"], n), "--", color=C_CAN, lw=1.2)
            ax.plot(*prepare_profile_for_plot(d["x_raw"], n), ":", color=C_RAW, lw=1.1)
            ax.plot(*prepare_profile_for_plot(d["z_full"], n), "-", color=C_HAD, lw=1.2)
            ax.set_ylim(min(d["z_full"].min(), d["z_can_matched"].min()) - 0.08, 1.28)
            if d["x_raw"][0] > 1.28:                     # segment-0 artefact leaves the axis: annotate it
                phi0 = physical_angles(n)[0]
                ax.annotate(f"segment 0: {d['x_raw'][0]:.1f}", xy=(phi0, 1.25), xytext=(phi0 + 30, 1.12),
                            fontsize=6.5, color=C_RAW, arrowprops=dict(arrowstyle="->", color=C_RAW, lw=0.7))
            ax.set_xticks(range(0, 361, 90)); ax.set_ylabel("Norm. intensity")
            ax.text(0.98, 0.04, f"NCC = {ncc(d['z_full'], d['z_can']):.3f}", transform=ax.transAxes, ha="right",
                    fontsize=7)
        axes[-1, 0].set_xlabel("Frame index k"); axes[-1, 1].set_xlabel("Hadamard index k")
        axes[-1, 2].set_xlabel("Angle $\\varphi$ (deg)")
        handles = [plt.Line2D([], [], color=C_CAN, marker="o", ms=2, label="canonical frames"),
                   plt.Line2D([], [], color=C_RAW, marker="s", ms=2, label="binary H$^+$ frames"),
                   plt.Line2D([], [], color="C3", marker="o", ls="", ms=3, label="$y_0$ (collects the frame offset)"),
                   plt.Line2D([], [], color="C0", marker="o", ls="", ms=3, label="$y_k$, $k \\geq 1$")]
        if show_corrected_coeffs:
            handles.append(plt.Line2D([], [], color="k", marker="_", ls="", ms=5, label="$y_k$ after offset correction"))
        handles += [plt.Line2D([], [], color=C_CAN, ls="--", label="canonical (scale and offset matched)"),
                    plt.Line2D([], [], color=C_RAW, ls=":", label="Hadamard, uncorrected"),
                    plt.Line2D([], [], color=C_HAD, ls="-", label=f"Hadamard, corrected ({correction_label})")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
        panel_labels(axes)
        savefig(fig, "fig03_full_basis_sampling", kind="article")
        plt.show()


def plot_dct_spectra(spectra, ns, roi_label):
    """spectra[(r, n)] = DCT coefficients s_k of the profile."""
    n_rois = len({r for r, _ in spectra})
    with article_style():
        fig, axes = plt.subplots(1, len(ns), figsize=(ART_W2, 2.0), squeeze=False, sharey=True, constrained_layout=True)
        for ax, n in zip(axes[0], ns):
            for r in range(n_rois):
                s = spectra[(r, n)]
                ax.semilogy(np.arange(1, n), np.abs(s[1:]) / np.abs(s[0]), "o-", ms=1.8, lw=0.9,
                            color=ROI_COLORS[r % 4], label=roi_label(r))
            ax.set_xlabel("DCT frequency index k"); ax.set_title(f"N = {n}"); ax.set_ylim(1e-4, 1)
        axes[0, 0].set_ylabel("$|s_k| / |s_0|$")
        axes[0, -1].legend(loc="upper right", fontsize=6.5)
        panel_labels(axes[0])
        savefig(fig, "fig04_dct_spectra", kind="article")
        plt.show()


def plot_orderings(patterns, coupling, n_vis, n_big):
    """patterns[name]: ordered +-1 Hadamard matrix (n_vis); coupling[name]: |Phi Psi| (n_big)."""
    with article_style():
        fig, axes = plt.subplots(2, len(patterns), figsize=(ART_W2, 4.3), constrained_layout=True)
        for j, name in enumerate(patterns):
            axes[0, j].imshow((patterns[name] + 1) / 2, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            axes[0, j].set_title(name)
            axes[0, j].set_xlabel("Segment j"); axes[0, j].set_ylabel("Measurement index")
            im = axes[1, j].imshow(coupling[name], cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
            axes[1, j].set_xlabel("DCT frequency index"); axes[1, j].set_ylabel("Measurement index")
            axes[1, j].set_title(f"$|\\Phi\\Psi|$, N = {n_big}")
        cb = fig.colorbar(im, ax=axes[1, :], fraction=0.025, pad=0.01); cb.ax.tick_params(labelsize=7)
        panel_labels(axes)
        savefig(fig, "fig05_orderings", kind="article")
        plt.show()


def plot_metric_vs_cr(df, group_col, metrics, ns, fname, colors=COLORS, kind="article"):
    """Mean (line) and standard deviation (band) over the ROIs, one row per N."""
    with article_style():
        fig, axes = plt.subplots(len(ns), len(metrics), figsize=(ART_W2, 1.6 * len(ns) + 0.45),
                                 squeeze=False, sharex=True, constrained_layout=True)
        for i, n in enumerate(ns):
            for j, (metric, label) in enumerate(metrics):
                ax = axes[i, j]
                for name, g in df[df["N"] == n].groupby(group_col, sort=False):
                    st = g.groupby("cr")[metric].agg(["mean", "std"]).reset_index()
                    c = colors.get(name, None)
                    ax.plot(st["cr"], st["mean"], "o-", ms=2, color=c, label=name)
                    ax.fill_between(st["cr"], st["mean"] - st["std"], st["mean"] + st["std"], color=c, alpha=0.15, lw=0)
                ax.set_ylabel(label); ax.grid(alpha=0.25, lw=0.4)
                if metric.startswith("ncc"):
                    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.02)
                if i == len(ns) - 1:
                    ax.set_xlabel("Compression ratio M/N")
            axes[i, 0].set_title(f"N = {n}", loc="left", fontweight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
        panel_labels(axes)
        savefig(fig, fname, kind=kind)
        plt.show()


def plot_plateau_best(data, selected, recon, ns, n_rois, roi_label):
    with article_style():
        fig, axes = plt.subplots(len(ns), n_rois, figsize=(ART_W2, 1.5 * len(ns) + 0.5), squeeze=False,
                                 sharex=True, sharey=True, constrained_layout=True)
        for i, n in enumerate(ns):
            for r in range(n_rois):
                ax, d = axes[i, r], data[(r, n)]
                ax.plot(*prepare_profile_for_plot(d["z_full"], n), color="black", lw=1.6)
                txt = []
                for _, row in selected[(selected["ROI"] == r) & (selected["N"] == n)].iterrows():
                    z = recon[(r, n, row["ordering"], row["requested_cr"])]
                    plateau = row["selection"] == "First plateau"
                    ax.plot(*prepare_profile_for_plot(z, n), color="#0072B2" if plateau else "#C00000",
                            ls="--" if plateau else "-.", lw=1.0)
                    txt.append(f"{'P' if plateau else 'B'}: {ABBR[row['ordering']]}, M = {row['M']}")
                ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=6)
                ax.set_xticks(range(0, 361, 90))
                if i == 0: ax.set_title(roi_label(r))
                if r == 0: ax.set_ylabel(f"N = {n}\nNorm. intensity")
                if i == len(ns) - 1: ax.set_xlabel("Angle $\\varphi$ (deg)")
        ax.set_ylim(None, 1.45)
        handles = [plt.Line2D([], [], color="black", lw=1.6, label="Hadamard, full basis (reference)"),
                   plt.Line2D([], [], color="#0072B2", ls="--", label="first plateau point (P)"),
                   plt.Line2D([], [], color="#C00000", ls="-.", label="best reconstruction (B)")]
        fig.legend(handles=handles, loc="outside upper center", ncol=3)
        panel_labels(axes)
        savefig(fig, "fig07_plateau_best_profiles", kind="article")
        plt.show()


def plot_reconstruction_grid(reference, recons, orderings, crs, n, title, fname, m_from_cr):
    """recons[(ordering, cr)] = reconstructed profile; reference = full Hadamard profile."""
    with article_style():
        fig, axes = plt.subplots(len(orderings), len(crs), figsize=(ART_W2, 0.95 * len(orderings) + 0.55),
                                 sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        for i, name in enumerate(orderings):
            for j, cr in enumerate(crs):
                ax = axes[i, j]
                ax.plot(*prepare_profile_for_plot(reference, n), color="0.7", lw=1.6)
                ax.plot(*prepare_profile_for_plot(recons[(name, cr)], n), color=COLORS[name], lw=1.0)
                ax.set_xticks([0, 180, 360])
                if i == 0: ax.set_title(f"CR = {int(round(cr * 100))}% (M = {m_from_cr(cr, n)})", fontsize=7)
                if j == 0: ax.set_ylabel(ABBR[name])
        fig.supxlabel("Angle $\\varphi$ (deg)", fontsize=8)
        fig.suptitle(title, fontsize=8.5)
        savefig(fig, fname, kind="article")
        plt.show()


def plot_noise_study(noise_df, regimes, n, title):
    """Diagnostic figure of the noise simulation (mean +- std over noise realisations)."""
    methods = noise_df["method"].unique()
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(regimes), 3, figsize=(17, 4.2 * len(regimes)), squeeze=False, sharex=True)
    for i, regime in enumerate(regimes):
        g = noise_df[noise_df["regime"] == regime]
        for j, (metric, label) in enumerate([("ncc", "NCC"), ("rmse", "RMSE"), ("direction_error", "Direction error (deg)")]):
            ax = axes[i, j]
            for k, m in enumerate(methods):
                st = g[g["method"] == m].groupby("photons")[metric].agg(["mean", "std"]).reset_index()
                ls = "--" if "BPDN" in m else ("-" if "full" in m else "-.")
                ax.errorbar(st["photons"], st["mean"], yerr=st["std"], color=cmap(k), ls=ls, marker="o", ms=4,
                            capsize=2, label=m)
            ax.set_xscale("log"); ax.set_ylabel(label); ax.grid(alpha=0.3)
            if metric == "rmse":
                ax.set_yscale("log")
            if i == len(regimes) - 1:
                ax.set_xlabel("Photo-electrons per segment and frame, P")
        axes[i, 0].set_title(regime, loc="left", fontweight="bold")
    axes[0, 2].legend(fontsize=9, loc="upper right")
    fig.suptitle(title, y=1.0)
    fig.tight_layout(); savefig(fig, "diag03_noise_robustness"); plt.show()


def plot_model_selection(cand_df, ns, adam_hist):
    """Diagnostic: validation NCC of closed form and Adam per training source; Adam training curves."""
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 3.8), squeeze=False, sharey=True)
    styles = {("hadamard", "closed form"): ("#C00000", "-", "o"), ("hadamard", "Adam"): ("#C00000", "--", "x"),
              ("canonical", "closed form"): ("#0072B2", "-", "o"), ("canonical", "Adam"): ("#0072B2", "--", "x")}
    for ax, n in zip(axes[0], ns):
        g = cand_df[(cand_df["N"] == n) & (cand_df["ordering"] == "Hadamard-Walsh")]
        for (src, meth), gg in g.groupby(["source", "method"]):
            st = gg.groupby("cr")["val_ncc"].max()
            c, ls, mk = styles.get((src, meth), ("k", "-", "o"))
            ax.plot(st.index, st.values, ls=ls, marker=mk, ms=5, color=c, label=f"{src}, {meth}")
        ax.set_title(f"N = {n}, Hadamard-Walsh"); ax.set_xlabel("Compression ratio M/N"); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Validation NCC (mean)"); axes[0, 0].legend(fontsize=10)
    fig.tight_layout(); savefig(fig, "diag04_learned_model_selection"); plt.show()
    if adam_hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (n, name, cr), h in adam_hist.items():
            if name == "Hadamard-Walsh" and cr in (0.125, 0.25, 0.5):
                ax.semilogy(h, label=f"N = {n}, CR = {cr}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss (standardised MSE)"); ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.3); fig.tight_layout(); savefig(fig, "diag05_adam_training_curves"); plt.show()


def plot_learned_vs_bp(dl_df, ns, orderings):
    with article_style():
        fig, axes = plt.subplots(len(ns), 2, figsize=(ART_W2, 1.6 * len(ns) + 0.45), squeeze=False, sharex=True,
                                 constrained_layout=True)
        for i, n in enumerate(ns):
            for j, (metric, label) in enumerate([("ncc", "NCC (mean)"), ("direction_error", "Dir. error (deg)")]):
                ax = axes[i, j]
                for name in orderings:
                    for method, ls, mk in [("Learned (best)", "-", "o"), ("SPGL1 BP", ":", "s")]:
                        g = dl_df[(dl_df["N"] == n) & (dl_df["ordering"] == name) & (dl_df["method"] == method)]
                        vals = g[metric] if metric == "ncc" else np.clip(g[metric], 0.1, None)  # log axis floor
                        ax.plot(g["cr"], vals, ls=ls, marker=mk, ms=2, color=COLORS[name])
                ax.set_ylabel(label); ax.grid(alpha=0.25, lw=0.4)
                if metric == "ncc":
                    ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1.02)
                else:
                    ax.set_yscale("log"); ax.set_ylim(0.08, 100)
            axes[i, 0].set_title(f"N = {n}", loc="left", fontweight="bold")
        for ax in axes[-1]:
            ax.set_xlabel("Compression ratio M/N")
        handles = ([plt.Line2D([], [], color=COLORS[nm], label=nm) for nm in orderings]
                   + [plt.Line2D([], [], color="k", ls="-", marker="o", ms=2, label="learned (best)"),
                      plt.Line2D([], [], color="k", ls=":", marker="s", ms=2, label="SPGL1 BP")])
        fig.legend(handles=handles, loc="outside upper center", ncol=5)
        panel_labels(axes)
        savefig(fig, "fig10_learned_vs_bp", kind="article")
        plt.show()


def direction_rgb(axis_deg, aniso, mask, vmax):
    """HSV image: hue = peak axis (0-180 deg), value = anisotropy / vmax, black outside the mask."""
    v = np.clip(aniso / vmax, 0, 1)
    rgb = mcolors.hsv_to_rgb(np.stack([axis_deg / 180.0, np.ones_like(v), v], axis=-1))
    rgb[~mask] = 0
    return rgb


def plot_direction_maps(maps, deltas, mask, vmax, delta_max=30.0):
    """
    maps[name] = (axis, A2) for the four maps (canonical, full Hadamard, BP, learned);
    deltas[name] = deviation map (deg, NaN outside the evaluated pixels) for canonical, BP and learned.
    """
    names = list(maps)
    with article_style():
        fig, axes = plt.subplots(2, 4, figsize=(ART_W2, 3.3), constrained_layout=True)
        for j, name in enumerate(names):
            axes[0, j].imshow(direction_rgb(*maps[name], mask, vmax), interpolation="nearest")
            axes[0, j].set_title(name, fontsize=7.5)
        cmap_d = plt.get_cmap("magma").copy(); cmap_d.set_bad("0.85")
        for j, name in {0: names[0], 2: names[2], 3: names[3]}.items():
            im_d = axes[1, j].imshow(deltas[name], cmap=cmap_d, vmin=0, vmax=delta_max, interpolation="nearest")
            axes[1, j].set_title(f"$\\Delta$, median = {np.nanmedian(deltas[name]):.1f}°", fontsize=7.5)
        axes[1, 1].axis("off")
        cax = axes[1, 1].inset_axes([0.1, 0.62, 0.8, 0.1])
        sm = plt.cm.ScalarMappable(cmap="hsv", norm=mcolors.Normalize(0, 180))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 45, 90, 135, 180])
        cb.set_label("peak axis $\\theta$ (deg), brightness: $A_2$", fontsize=7); cb.ax.tick_params(labelsize=6.5)
        cax2 = axes[1, 1].inset_axes([0.1, 0.22, 0.8, 0.1])
        cb2 = fig.colorbar(im_d, cax=cax2, orientation="horizontal")
        cb2.set_label("$\\Delta$ to full Hadamard (deg)", fontsize=7); cb2.ax.tick_params(labelsize=6.5)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        panel_labels([axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 2], axes[1, 3]])
        savefig(fig, "fig11_direction_maps", kind="article")
        plt.show()


def format_duration(seconds):
    """Human-readable duration."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    if seconds < 2 * 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"
