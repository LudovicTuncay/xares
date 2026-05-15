from __future__ import annotations

import argparse
import ast
import io
import json
import math
import random
import re
import sqlite3
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm



AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".wma",
}

DOMAIN_ORDER = ["sounds", "music", "speech"]


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    display_name: str
    domain: str
    env_dir: Path
    tar_paths: list[Path]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def parse_readme_domains(readme_path: Path) -> tuple[dict[str, str], set[str]]:
    """Returns (normalized_name->domain, normalized_name set found in README)."""

    text = readme_path.read_text(encoding="utf-8")

    domain_sections: dict[str, str] = {
        "Speech": "speech",
        "Environment": "sounds",
        "Music": "music",
    }

    name_to_domain: dict[str, str] = {}
    readme_names: set[str] = set()

    for heading, domain in domain_sections.items():
        m = re.search(rf"^###\s+{re.escape(heading)}\s*$", text, flags=re.MULTILINE)
        if not m:
            continue
        section_start = m.end()

        next_heading = re.search(r"^###\s+", text[section_start:], flags=re.MULTILINE)
        section_end = section_start + (next_heading.start() if next_heading else len(text) - section_start)
        section = text[section_start:section_end]

        for line in section.splitlines():
            item = re.match(r"^\s*-\s*\[[xX \t]\]\s*(.+?)\s*$", line)
            if not item:
                continue
            raw_name = item.group(1)
            norm = _normalize_name(raw_name)
            name_to_domain[norm] = domain
            readme_names.add(norm)

    # Also collect task names mentioned in the result tables.
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if not parts or parts[0] in {"Task", ":------------------------------:"}:
            continue
        readme_names.add(_normalize_name(parts[0]))

    return name_to_domain, readme_names


def load_task_name_mapping(tasks_dir: Path) -> dict[str, str]:
    """Parse task configs to get dataset_id -> formal_name."""

    mapping: dict[str, str] = {}

    for task_file in sorted(tasks_dir.glob("*_task.py")):
        if task_file.name == "__init__.py":
            continue
        tree = ast.parse(task_file.read_text(encoding="utf-8"), filename=str(task_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func_name: str | None = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name != "TaskConfig":
                continue

            dataset_id: str | None = None
            formal_name: str | None = None
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    dataset_id = kw.value.value
                if (
                    kw.arg == "formal_name"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    formal_name = kw.value.value

            if dataset_id:
                mapping[dataset_id] = formal_name or dataset_id

    return mapping


def infer_domain(dataset_id: str, display_name: str, name_to_domain: dict[str, str]) -> str:
    norm = _normalize_name(display_name)
    if norm in name_to_domain:
        return name_to_domain[norm]

    lower = f"{dataset_id} {display_name}".lower()
    if any(k in lower for k in ["speech", "libri", "voxceleb", "voxlingua", "asv", "asvspoof", "crema", "ravdess", "fluent", "vocal"]):
        return "speech"
    if any(k in lower for k in ["music", "maestro", "nsynth", "gtzan", "fma", "freemusic"]):
        return "music"
    return "sounds"


def discover_downloaded_datasets(
    env_root: Path,
    dataset_id_to_name: dict[str, str],
    name_to_domain: dict[str, str],
    readme_names: set[str],
) -> list[DatasetInfo]:
    datasets: list[DatasetInfo] = []

    for env_dir in sorted(p for p in env_root.iterdir() if p.is_dir()):
        tar_paths = sorted(
            p
            for p in env_dir.glob("*.tar")
            if p.is_file() and p.stat().st_size > 0 and "encoded" not in p.name.lower()
        )
        if not tar_paths:
            continue

        dataset_id = env_dir.name
        display_name = dataset_id_to_name.get(dataset_id, dataset_id)
        domain = infer_domain(dataset_id, display_name, name_to_domain)

        if _normalize_name(display_name) not in readme_names:
            logger.warning(
                f"Dataset '{display_name}' not found in README; keeping task config name."
            )

        datasets.append(
            DatasetInfo(
                dataset_id=dataset_id,
                display_name=display_name,
                domain=domain,
                env_dir=env_dir,
                tar_paths=tar_paths,
            )
        )

    # Stable ordering: domain then name
    datasets.sort(key=lambda d: (DOMAIN_ORDER.index(d.domain) if d.domain in DOMAIN_ORDER else 999, d.display_name))
    return datasets


def iter_audio_members(tar: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for member in tar:
        if not member.isfile():
            continue
        ext = Path(member.name).suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            yield member


def count_audio_files_by_shard(tar_paths: list[Path]) -> tuple[int, dict[str, int]]:
    total = 0
    by_shard: dict[str, int] = {}
    for tar_path in tar_paths:
        with tarfile.open(tar_path, mode="r:*") as tar:
            n = sum(1 for _ in iter_audio_members(tar))
            by_shard[str(tar_path)] = n
            total += n
    return total, by_shard


def count_audio_files(tar_paths: list[Path]) -> int:
    total, _ = count_audio_files_by_shard(tar_paths)
    return total


def _reservoir_slot(i: int, k: int, rng: random.Random) -> int | None:
    if k <= 0:
        return None
    if i < k:
        return i
    j = rng.randint(0, i)
    if j < k:
        return j
    return None


def audio_metadata_from_bytes(data: bytes, suffix: str) -> tuple[float, int, int] | None:
    """Returns (duration_seconds, sample_rate, num_frames) if available."""

    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if f.samplerate <= 0:
                return None
            num_frames = int(len(f))
            sample_rate = int(f.samplerate)
            duration_s = float(num_frames) / float(sample_rate)
            return duration_s, sample_rate, num_frames
    except Exception:
        pass

    try:
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            info = torchaudio.info(tmp.name)
            if info.sample_rate <= 0:
                return None
            num_frames = int(info.num_frames)
            sample_rate = int(info.sample_rate)
            duration_s = float(num_frames) / float(sample_rate)
            return duration_s, sample_rate, num_frames
    except Exception:
        return None


def compute_dataset_durations(
    dataset: DatasetInfo,
    n_audio: int,
    max_per_dataset: int,
    seed: int,
    global_audio_pbar: tqdm,
) -> tuple[list[float], int]:
    dataset_seed = seed + zlib.crc32(dataset.dataset_id.encode("utf-8"))
    rng = random.Random(dataset_seed)

    sample_cap = min(max_per_dataset, n_audio)
    sampled_durations: list[float | None] = [None] * sample_cap

    failures = 0
    audio_index = -1

    with tqdm(total=n_audio, desc=dataset.display_name, leave=False, unit="clip") as pbar:
        for tar_path in dataset.tar_paths:
            with tarfile.open(tar_path, mode="r:*") as tar:
                for member in iter_audio_members(tar):
                    audio_index += 1
                    slot = _reservoir_slot(audio_index, sample_cap, rng)

                    if slot is not None:
                        f = tar.extractfile(member)
                        if f is None:
                            failures += 1
                        else:
                            data = f.read()
                            suffix = Path(member.name).suffix.lstrip(".").lower()
                            meta = audio_metadata_from_bytes(data, suffix=suffix)
                            if meta is None:
                                failures += 1
                            else:
                                duration_s, _sr, _frames = meta
                                if not math.isfinite(duration_s) or duration_s <= 0:
                                    failures += 1
                                else:
                                    sampled_durations[slot] = duration_s

                    pbar.update(1)
                    global_audio_pbar.update(1)

    durations = [d for d in sampled_durations if d is not None]
    return durations, failures


def init_manifest_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_files (
            dataset_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            domain TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            member_name TEXT NOT NULL,
            key TEXT,
            ext TEXT,
            size_bytes INTEGER,
            sample_rate INTEGER,
            num_frames INTEGER,
            duration_s REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (dataset_id, shard_path, member_name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audio_files_dataset ON audio_files(dataset_id)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_shards (
            dataset_id TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            mtime_ns INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            processed_at TEXT NOT NULL,
            PRIMARY KEY (dataset_id, shard_path)
        )
        """
    )

    return conn


def shard_is_processed(
    conn: sqlite3.Connection,
    dataset_id: str,
    tar_path: Path,
) -> bool:
    st = tar_path.stat()
    row = conn.execute(
        "SELECT mtime_ns, size_bytes FROM processed_shards WHERE dataset_id=? AND shard_path=?",
        (dataset_id, str(tar_path)),
    ).fetchone()
    if row is None:
        return False
    return int(row[0]) == int(st.st_mtime_ns) and int(row[1]) == int(st.st_size)


def mark_shard_processed(conn: sqlite3.Connection, dataset_id: str, tar_path: Path) -> None:
    st = tar_path.stat()
    conn.execute(
        """
        INSERT OR REPLACE INTO processed_shards(dataset_id, shard_path, mtime_ns, size_bytes, processed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            str(tar_path),
            int(st.st_mtime_ns),
            int(st.st_size),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def build_audio_manifest(
    *,
    conn: sqlite3.Connection,
    datasets: list[DatasetInfo],
    counts: dict[str, int],
    counts_by_shard: dict[str, dict[str, int]],
    resume: bool,
) -> None:
    total_audio = sum(counts.values())

    with tqdm(total=len(datasets), desc="Manifest datasets", unit="dataset") as dataset_pbar:
        with tqdm(total=total_audio, desc="Manifest audio", unit="clip") as audio_pbar:
            for dataset in datasets:
                dataset_total = counts[dataset.dataset_id]
                dataset_counts_by_shard = counts_by_shard.get(dataset.dataset_id, {})

                with tqdm(total=dataset_total, desc=dataset.display_name, leave=False, unit="clip") as ds_pbar:
                    for tar_path in dataset.tar_paths:
                        shard_count = dataset_counts_by_shard.get(str(tar_path), 0)
                        if resume and shard_is_processed(conn, dataset.dataset_id, tar_path):
                            ds_pbar.update(shard_count)
                            audio_pbar.update(shard_count)
                            continue

                        batch: list[tuple] = []
                        with tarfile.open(tar_path, mode="r:*") as tar:
                            for member in iter_audio_members(tar):
                                f = tar.extractfile(member)
                                if f is None:
                                    ds_pbar.update(1)
                                    audio_pbar.update(1)
                                    continue

                                data = f.read()
                                suffix = Path(member.name).suffix.lstrip(".").lower()
                                meta = audio_metadata_from_bytes(data, suffix=suffix)
                                if meta is None:
                                    sample_rate = None
                                    num_frames = None
                                    duration_s = None
                                else:
                                    duration_s, sample_rate, num_frames = meta

                                member_path = Path(member.name)
                                key = member_path.stem

                                batch.append(
                                    (
                                        dataset.dataset_id,
                                        dataset.display_name,
                                        dataset.domain,
                                        str(tar_path),
                                        member.name,
                                        key,
                                        member_path.suffix.lower(),
                                        int(member.size),
                                        sample_rate,
                                        num_frames,
                                        duration_s,
                                        datetime.now(timezone.utc).isoformat(),
                                    )
                                )

                                if len(batch) >= 500:
                                    conn.executemany(
                                        """
                                        INSERT OR IGNORE INTO audio_files(
                                            dataset_id, display_name, domain, shard_path, member_name,
                                            key, ext, size_bytes, sample_rate, num_frames, duration_s, created_at
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        batch,
                                    )
                                    conn.commit()
                                    batch.clear()

                                ds_pbar.update(1)
                                audio_pbar.update(1)

                        if batch:
                            conn.executemany(
                                """
                                INSERT OR IGNORE INTO audio_files(
                                    dataset_id, display_name, domain, shard_path, member_name,
                                    key, ext, size_bytes, sample_rate, num_frames, duration_s, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                batch,
                            )
                            conn.commit()

                        mark_shard_processed(conn, dataset.dataset_id, tar_path)
                        conn.commit()

                dataset_pbar.update(1)


def iter_manifest_durations(
    conn: sqlite3.Connection,
    dataset_id: str,
) -> Iterable[float]:
    cur = conn.execute(
        "SELECT duration_s FROM audio_files WHERE dataset_id=? AND duration_s IS NOT NULL",
        (dataset_id,),
    )
    for (duration_s,) in cur:
        if duration_s is None:
            continue
        d = float(duration_s)
        if math.isfinite(d) and d > 0:
            yield d


def load_stats_from_manifest(
    *,
    conn: sqlite3.Connection,
    datasets: list[DatasetInfo],
    max_per_dataset: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []

    dataset_counts: dict[str, int] = {}
    for d in datasets:
        (n_audio,) = conn.execute(
            "SELECT COUNT(*) FROM audio_files WHERE dataset_id=?",
            (d.dataset_id,),
        ).fetchone()
        dataset_counts[d.dataset_id] = int(n_audio)

    total_audio = sum(dataset_counts.values())

    with tqdm(total=len(datasets), desc="Datasets", unit="dataset") as dataset_pbar:
        with tqdm(total=total_audio, desc="Audio (manifest)", unit="clip") as audio_pbar:
            for d in datasets:
                n_audio = dataset_counts[d.dataset_id]
                dataset_seed = seed + zlib.crc32(d.dataset_id.encode("utf-8"))
                rng = random.Random(dataset_seed)

                sample_cap = min(max_per_dataset, n_audio)
                sampled: list[float | None] = [None] * sample_cap
                seen = -1

                for duration_s in iter_manifest_durations(conn, d.dataset_id):
                    seen += 1
                    slot = _reservoir_slot(seen, sample_cap, rng)
                    if slot is not None:
                        sampled[slot] = duration_s
                    audio_pbar.update(1)

                durations_s = [x for x in sampled if x is not None]
                summary = summarize_durations(durations_s)

                rows.append(
                    {
                        "dataset_id": d.dataset_id,
                        "display_name": d.display_name,
                        "domain": d.domain,
                        "env_dir": str(d.env_dir),
                        "n_audio": int(n_audio),
                        "n_failures": 0,
                        "sample_cap": int(max_per_dataset),
                        "sampled": bool(n_audio > max_per_dataset),
                        "durations_s": durations_s,
                        **summary,
                    }
                )
                dataset_pbar.update(1)

    return rows


def summarize_durations(durations_s: list[float]) -> dict[str, float | int]:
    if not durations_s:
        return {
            "n_measured": 0,
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "p05_s": float("nan"),
            "p25_s": float("nan"),
            "p75_s": float("nan"),
            "p95_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
        }

    arr = np.asarray(durations_s, dtype=np.float64)
    q05, q25, q75, q95 = np.quantile(arr, [0.05, 0.25, 0.75, 0.95]).tolist()
    return {
        "n_measured": int(arr.size),
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p05_s": float(q05),
        "p25_s": float(q25),
        "p75_s": float(q75),
        "p95_s": float(q95),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def draw_density_curve(
    ax,
    row: dict,
    y_base: float,
    scale: float,
    x_min: float,
    x_max: float,
    bins: np.ndarray,
    x_centers: np.ndarray,
    kernel: np.ndarray,
    domain_colors: dict,
) -> None:
    durations = np.asarray(row["durations_s"], dtype=np.float64)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    
    x_range = x_max - x_min
    
    # Check for constant or near-constant length (low variance)
    # If range < 1ms or std dev is tiny, treat as constant.
    is_constant = False
    if len(durations) > 1:
        if durations.max() - durations.min() < 0.01:
            is_constant = True
    elif len(durations) == 1:
        is_constant = True
        
    domain = row["domain"]
    base_color = domain_colors.get(domain, (0.4, 0.4, 0.4))
    
    if is_constant:
        # Draw a vertical line/bar for constant length
        val = float(np.mean(durations))
        
        # We can't draw a density curve. We draw a stylized marker.
        # Vertical line from y_base to y_base + scale
        ax.plot([val, val], [y_base, y_base + scale * 0.8], color=base_color, linewidth=2.5, alpha=0.8)
        
        # Add a cap or dot
        ax.scatter([val], [y_base + scale * 0.8], color=base_color, s=20)
        
        # No quantile bars needed really, or just a single tick
        # We'll just leave it as the vertical indicator.
        
    else:
        # Standard KDE/Histogram
        # Filter for histogram to avoid RuntimeWarning with density=True if range is empty
        durations_in_range = durations[(durations >= x_min) & (durations <= x_max)]
        if len(durations_in_range) > 0:
            hist, _ = np.histogram(durations_in_range, bins=bins, density=True)
        else:
            hist = np.zeros(len(bins) - 1)

        smooth = np.convolve(hist, kernel, mode="same")
        if smooth.max() > 0:
            smooth = smooth / smooth.max()

        # Transparent fill under curve
        ax.fill_between(x_centers, y_base, y_base + smooth * scale, color=base_color, alpha=0.15, linewidth=0)
        
        # Outline
        ax.plot(x_centers, y_base + smooth * scale, color=base_color, linewidth=1.3)

        # Quantile BAR below the distribution
        q_levels = [0.025, 0.10, 0.25, 0.75, 0.90, 0.975]
        qs = np.quantile(durations, q_levels)
        
        # Bar geometry: placed just below y_base
        bar_height = 0.08
        bar_y_bottom = y_base - bar_height
        bar_y_top = y_base
        
        # Enforce min width for visibility (0.8% of range)
        min_w = 0.008 * x_range
        
        def draw_bar_rect(q_start, q_end, color):
            w = q_end - q_start
            if w < min_w:
                center = (q_start + q_end) / 2
                q_start = center - min_w / 2
                q_end = center + min_w / 2
            ax.fill_between([q_start, q_end], bar_y_bottom, bar_y_top, color=color, linewidth=0)
        
        # Full range (2.5 - 97.5)
        draw_bar_rect(qs[0], qs[5], "#E5E7EB") # gray-200
        # Mid range (10 - 90)
        draw_bar_rect(qs[1], qs[4], "#9CA3AF") # gray-400
        # Core range (25 - 75)
        draw_bar_rect(qs[2], qs[3], "#4B5563") # gray-600

        # Mean marker
        mean_value = float(np.mean(durations))
        
        # Dot on the curve (Removed as per request)
        # curve_y = y_base
        # if x_min <= mean_value <= x_max:
        #      idx = int(np.clip(np.interp(mean_value, x_centers, np.arange(len(x_centers))), 0, len(smooth)-1))
        #      curve_y = y_base + smooth[idx] * scale

        # ax.scatter(
        #     [mean_value],
        #     [curve_y],
        #     color="black",
        #     s=10,
        #     zorder=3,
        # )
        
        # Vertical line on bar
        ax.plot([mean_value, mean_value], [bar_y_bottom, bar_y_top], color="white", linewidth=1.5, zorder=2)


def format_duration_val(s: float) -> str:
    if s >= 60:
        val = s / 60
        if abs(val - round(val)) < 0.1:
            return f"{int(round(val))}m"
        return f"{val:.1f}m"
    if abs(s - round(s)) < 0.1:
        return f"{int(round(s))}s"
    return f"{s:.1f}s"


def plot_ridgeline(
    rows: list[dict],
    out_svg: Path,
    title: str,
    plot_mode: str = "global",
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install with `uv sync --extra viz` (or `uv sync --all-extras`)."
        ) from e

    rows = [r for r in rows if r.get("n_measured", 0) and math.isfinite(r.get("median_s", float("nan")))]
    if not rows:
        logger.warning("No duration samples available; skipping plot.")
        return

    # Domain-first ordering, then by median duration.
    rows.sort(
        key=lambda r: (
            DOMAIN_ORDER.index(r["domain"]) if r["domain"] in DOMAIN_ORDER else 999,
            r["median_s"],
        )
    )

    all_durations = np.concatenate([np.asarray(r["durations_s"], dtype=np.float64) for r in rows])
    all_durations = all_durations[np.isfinite(all_durations) & (all_durations > 0)]

    x_min = float(np.quantile(all_durations, 0.01))
    x_max = float(np.quantile(all_durations, 0.99))
    x_pad = 0.08 * (x_max - x_min + 1e-6)
    x_min = max(0.0, x_min - x_pad)
    x_max = x_max + x_pad

    x_label_pad = 0.08 * (x_max - x_min + 1e-6)
    x_max_plot = x_max + x_label_pad

    bins = np.linspace(x_min, x_max, 250)
    x_centers = (bins[:-1] + bins[1:]) / 2.0

    domain_colors = {
        "music": mcolors.to_rgb("#4F6D9A"),   # muted blue
        "speech": mcolors.to_rgb("#4C9A8B"),  # muted teal
        "sounds": mcolors.to_rgb("#C7A164"),  # muted sand
    }
    
    # Gaussian smoothing kernel (in bin units)
    sigma = 1.7
    half = int(4 * sigma)
    kx = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (kx / sigma) ** 2)
    kernel /= kernel.sum()

    # Common styling
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "axes.grid": False,
            "axes.axisbelow": True,
            "font.size": 8, # Default size 8pt
            "font.family": "DejaVu Sans Mono", # Font family
            "svg.fonttype": "none", # Text as text, not paths
        }
    )
    
    def setup_axis(ax_obj):
        # Dashed grid lines
        ax_obj.grid(axis="x", color="#E5E7EB", linewidth=0.8, linestyle="--")
        ax_obj.set_yticks([])
        for spine in ["top", "right", "left", "bottom"]:
            ax_obj.spines[spine].set_visible(False)
        
        # Custom X ticks
        def format_tick_seconds(seconds: float) -> str:
            if seconds == 0:
                return "0"
            if seconds >= 60 and abs(seconds / 60 - round(seconds / 60)) < 1e-6:
                return f"{int(round(seconds / 60))}m"
            if abs(seconds - round(seconds)) < 1e-6:
                 val = int(round(seconds))
                 if val == 30:
                     return "30s"
                 return f"{val}"
            return f"{seconds:g}"

        tick_seconds = np.array([0, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800], dtype=np.float64)
        keep = (tick_seconds >= x_min) & (tick_seconds <= x_max)
        tick_seconds = tick_seconds[keep]
        
        # Explicitly set ticks to valid positive seconds only
        # This prevents grid lines from appearing in the negative margin area (used for labels)
        ax_obj.set_xticks(tick_seconds)
        ax_obj.set_xticklabels([]) # Hide default labels (we draw them manually)
        
        return tick_seconds, format_tick_seconds

    if plot_mode in ["global", "both"]:
        n = len(rows)
        fig_h = max(4.5, 0.36 * n + 1.6)
        
        # 5.9 inches width for A4 with margins
        fig, ax = plt.subplots(figsize=(5.9, fig_h), constrained_layout=True)
        tick_seconds, fmt_func = setup_axis(ax)
        
        y_positions = np.arange(n)
        scale = 0.9
        x_range = x_max - x_min
        
        for i, (y, row) in enumerate(zip(y_positions, rows)):
            draw_density_curve(ax, row, float(y), scale, x_min, x_max, bins, x_centers, kernel, domain_colors)

            # Labels
            ax.text(
                x_min - 0.02 * x_range,
                y + 0.1,
                row["display_name"].upper(),
                ha="right",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#111827",
                transform=ax.transData
            )

        # Global ticks at bottom
        tick_label_y = -0.62
        for tick in tick_seconds:
            ax.text(
                tick,
                tick_label_y,
                fmt_func(float(tick)),
                ha="center",
                va="top",
                fontsize=8,
                color="#111827",
            )
            
        # Left margin to be 1/3 of total width. Plot is 2/3.
        # Margin width = 0.5 * Plot width
        x_left_margin = x_min - 0.5 * x_range
        ax.set_xlim(x_left_margin, x_max_plot)

        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg)
        plt.close(fig)
    
    if plot_mode in ["individual", "both"]:
        # Output directory based on prefix
        out_dir = out_svg.parent
        
        for row in rows:
            # Short figure height (one row), width 5.9 inches
            # User requested thinner figures (1 to 1.2 inches)
            fig, ax = plt.subplots(figsize=(5.9, 1.1), constrained_layout=True)
            tick_seconds, fmt_func = setup_axis(ax)
            
            x_range = x_max - x_min
            
            draw_density_curve(ax, row, 0.0, 0.9, x_min, x_max, bins, x_centers, kernel, domain_colors)
            
            # Labels (same positions relative to y=0)
            ax.text(
                x_min - 0.02 * x_range,
                0.1,
                row["display_name"].upper(),
                ha="right",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#111827",
                transform=ax.transData
            )
            
            # Ticks (drawn manually since we stripped axes)
            tick_y = -0.15
            for tick in tick_seconds:
                # Add small tick mark?
                ax.plot([tick, tick], [0, -0.05], color="#E5E7EB", linewidth=1)
                ax.text(
                    tick,
                    tick_y,
                    fmt_func(float(tick)),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#111827",
                )
            
            # Left margin to be 1/3 of total width. Plot is 2/3.
            # Margin width = 0.5 * Plot width
            x_left_margin = x_min - 0.5 * x_range
            ax.set_xlim(x_left_margin, x_max_plot)
            ax.set_ylim(-0.3, 1.2) # Give space for ticks
            
            safe_name = _normalize_name(row["dataset_id"])
            p_svg = out_dir / f"{out_svg.stem}_{safe_name}.svg"
            
            fig.savefig(p_svg)
            plt.close(fig)




def write_csv(rows: list[dict], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "display_name",
        "domain",
        "env_dir",
        "n_audio",
        "n_measured",
        "n_failures",
        "sample_cap",
        "sampled",
        "mean_s",
        "median_s",
        "p05_s",
        "p25_s",
        "p75_s",
        "p95_s",
        "min_s",
        "max_s",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe downloaded ./env datasets (counts + length distributions).")
    parser.add_argument("--env-root", type=str, default=None, help="Env root dir (default: ./env)")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Dataset ids to include (env folder names)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Dataset ids to exclude (env folder names)",
    )
    parser.add_argument("--max-per-dataset", type=int, default=50_000, help="Max clips per dataset for durations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--out-csv", type=str, default="csv_results/dataset_stats.csv", help="Output CSV path")
    parser.add_argument(
        "--out-fig-prefix",
        type=str,
        default="figures/dataset_length_ridgeline",
        help="Output figure prefix (writes .pdf and .png)",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["global", "individual", "both"],
        default="global",
        help="Plot mode: 'global' (one large ridgeline), 'individual' (one fig per dataset), or 'both'.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--dump-json", type=str, default=None, help="Optional path to dump raw summary JSON")
    parser.add_argument(
        "--manifest-sqlite",
        type=str,
        default=None,
        help="SQLite file to store per-audio metadata (single-file manifest)",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Scan all audio and write/update the manifest SQLite",
    )
    parser.add_argument(
        "--resume-manifest",
        action="store_true",
        help="Skip shards already recorded in manifest (default on)",
    )
    parser.set_defaults(resume_manifest=True)
    parser.add_argument(
        "--no-resume-manifest",
        action="store_false",
        dest="resume_manifest",
        help="Disable manifest resume; reprocess all shards",
    )
    parser.add_argument(
        "--use-manifest",
        action="store_true",
        help="Compute stats/plot from manifest (no tar reads)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="After building manifest, stop without plotting/stats",
    )
    args = parser.parse_args()

    env_root = Path(args.env_root) if args.env_root else Path("./env")

    readme_path = Path("README.md")
    name_to_domain, readme_names = parse_readme_domains(readme_path)

    dataset_id_to_name = load_task_name_mapping(Path("src/tasks"))

    datasets = discover_downloaded_datasets(env_root, dataset_id_to_name, name_to_domain, readme_names)

    if args.include:
        include_set = set(args.include)
        datasets = [d for d in datasets if d.dataset_id in include_set]

    if args.exclude:
        exclude_set = set(args.exclude)
        datasets = [d for d in datasets if d.dataset_id not in exclude_set]

    if not datasets:
        logger.error(f"No downloaded datasets found under {env_root} (after filters).")
        return 2

    logger.info(f"Found {len(datasets)} downloaded datasets under {env_root}.")

    manifest_path = Path(args.manifest_sqlite) if args.manifest_sqlite else None
    manifest_conn: sqlite3.Connection | None = None

    if args.build_manifest:
        if manifest_path is None:
            logger.error("--build-manifest requires --manifest-sqlite PATH")
            return 2
        manifest_conn = init_manifest_db(manifest_path)

    if args.use_manifest:
        if manifest_path is None:
            logger.error("--use-manifest requires --manifest-sqlite PATH")
            return 2
        if not manifest_path.exists():
            logger.error(f"Manifest file not found: {manifest_path}")
            return 2
        if manifest_conn is None:
            manifest_conn = init_manifest_db(manifest_path)

    counts: dict[str, int] = {}
    counts_by_shard: dict[str, dict[str, int]] = {}

    if args.use_manifest and not args.build_manifest:
        assert manifest_conn is not None
        rows = load_stats_from_manifest(
            conn=manifest_conn,
            datasets=datasets,
            max_per_dataset=args.max_per_dataset,
            seed=args.seed,
        )
    else:
        # Count audio files first, to make progress bars meaningful.
        logger.info("Counting audio clips per dataset...")
        for d in tqdm(datasets, desc="Counting", unit="dataset"):
            total, by_shard = count_audio_files_by_shard(d.tar_paths)
            counts[d.dataset_id] = total
            counts_by_shard[d.dataset_id] = by_shard

        total_audio = sum(counts.values())
        logger.info(f"Total audio clips (all datasets): {total_audio:,}")

        if args.build_manifest:
            assert manifest_conn is not None
            build_audio_manifest(
                conn=manifest_conn,
                datasets=datasets,
                counts=counts,
                counts_by_shard=counts_by_shard,
                resume=bool(args.resume_manifest),
            )
            logger.info(f"Wrote/updated manifest SQLite at {manifest_path}.")

            if args.manifest_only:
                return 0

            if args.use_manifest:
                assert manifest_conn is not None
                rows = load_stats_from_manifest(
                    conn=manifest_conn,
                    datasets=datasets,
                    max_per_dataset=args.max_per_dataset,
                    seed=args.seed,
                )
            else:
                rows = []
                with tqdm(total=len(datasets), desc="Datasets", unit="dataset") as dataset_pbar:
                    with tqdm(total=total_audio, desc="Audio scanned", unit="clip") as global_audio_pbar:
                        for d in datasets:
                            n_audio = counts[d.dataset_id]
                            durations_s, failures = compute_dataset_durations(
                                dataset=d,
                                n_audio=n_audio,
                                max_per_dataset=args.max_per_dataset,
                                seed=args.seed,
                                global_audio_pbar=global_audio_pbar,
                            )

                            summary = summarize_durations(durations_s)
                            rows.append(
                                {
                                    "dataset_id": d.dataset_id,
                                    "display_name": d.display_name,
                                    "domain": d.domain,
                                    "env_dir": str(d.env_dir),
                                    "n_audio": int(n_audio),
                                    "n_failures": int(failures),
                                    "sample_cap": int(args.max_per_dataset),
                                    "sampled": bool(n_audio > args.max_per_dataset),
                                    "durations_s": durations_s,
                                    **summary,
                                }
                            )

                            dataset_pbar.update(1)
        else:
            rows = []
            with tqdm(total=len(datasets), desc="Datasets", unit="dataset") as dataset_pbar:
                with tqdm(total=total_audio, desc="Audio scanned", unit="clip") as global_audio_pbar:
                    for d in datasets:
                        n_audio = counts[d.dataset_id]
                        durations_s, failures = compute_dataset_durations(
                            dataset=d,
                            n_audio=n_audio,
                            max_per_dataset=args.max_per_dataset,
                            seed=args.seed,
                            global_audio_pbar=global_audio_pbar,
                        )

                        summary = summarize_durations(durations_s)
                        rows.append(
                            {
                                "dataset_id": d.dataset_id,
                                "display_name": d.display_name,
                                "domain": d.domain,
                                "env_dir": str(d.env_dir),
                                "n_audio": int(n_audio),
                                "n_failures": int(failures),
                                "sample_cap": int(args.max_per_dataset),
                                "sampled": bool(n_audio > args.max_per_dataset),
                                "durations_s": durations_s,
                                **summary,
                            }
                        )

                        dataset_pbar.update(1)

    out_csv = Path(args.out_csv)
    write_csv(rows, out_csv)
    logger.info(f"Wrote stats CSV to {out_csv}.")

    if args.dump_json:
        out_json = Path(args.dump_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for r in rows:
            rr = dict(r)
            rr["durations_s"] = rr.get("durations_s", [])
            payload.append(rr)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Wrote JSON dump to {out_json}.")

    if not args.no_plot:
        prefix = Path(args.out_fig_prefix)
        plot_ridgeline(
            rows=rows,
            out_svg=prefix.with_suffix(".svg"),
            title="Audio length distributions across X-ARES datasets",
            plot_mode=args.plot_mode,
        )
        if args.plot_mode == "global":
             logger.info(f"Wrote figure to {prefix.with_suffix('.svg')}.")
        else:
             logger.info(f"Wrote figures to {prefix.parent} (mode={args.plot_mode}).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
