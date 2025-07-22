import torch
from torchaudio.compliance.kaldi import fbank


def normalize_audio(wav: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize audio waveform to be between -1 and 1.
    """
    wav = wav - torch.mean(wav)
    return wav / (torch.max(torch.abs(wav)) + eps) # added eps to avoid division by zero


# ────────────────────────────────────────────────────────────────────────────────
# Helper ─ compute the hop (H) and frame length (L) that give the wanted
# number of frames, while preserving  L = ratio · H
# ────────────────────────────────────────────────────────────────────────────────
def hop_and_frame_for_target(
        num_samples: int,
        sr: int,
        target_frames: int,
        ratio: float = 2.5,          # L / H
        snip_edges: bool = True,
):
    """
    Return (frame_ms, hop_ms) - both in milliseconds - so that
    fbank(..., frame_length=frame_ms, frame_shift=hop_ms, ...) yields
    `target_frames` time bins on a signal of `num_samples` samples.
    """
    if snip_edges:
        # Kaldi's default: keep only full frames
        H_float = num_samples / (target_frames - 1 + ratio)
    else:
        # Kaldi-style symmetric extension at the edges
        H_float = num_samples / target_frames

    H = max(1, int(round(H_float)))        # hop in *samples*
    L = int(round(ratio * H))              # frame length in *samples*
    hop_ms   = 1_000.0 * H / sr            # → ms
    frame_ms = 1_000.0 * L / sr
    return frame_ms, hop_ms


# ────────────────────────────────────────────────────────────────────────────────
# Module that always hits the requested number of time bins
# ────────────────────────────────────────────────────────────────────────────────
class MelSpecTransform(torch.nn.Module):
    """
    Compute a Kaldi-compatible log-mel filter-bank spectrogram.

    Parameters
    ----------
    sr : int
        Sample rate (Hz).
    n_mels : int
        Number of mel bands.
    clip_length : float
        Length of the input clip in *seconds*.
        (Used only when hop_length and/or frame_length are not supplied.)
    target_time_bins : int, default 1024
        Desired number of time frames in the output.
    ratio : float, default 2.5
        Frame-length-to-hop ratio (L / H).
    frame_length : float or None, default None
        Analysis window length in *milliseconds*.
    hop_length : float or None, default None
        Hop (frame shift) in *milliseconds*.
    f_min, f_max : int
        Frequency range for the mel filter bank.
    log : bool, default True
        Return log-amplitude FBanks (Kaldi default).
    snip_edges : bool, default True
        Match Kaldi's frame-counting rule.
    """
    def __init__(
        self,
        sr: int,
        n_mels: int,
        clip_length: float,
        target_time_bins: int = 256,
        ratio: float = 2.5,
        frame_length: float | None = None,   # ms
        hop_length: float | None = None,     # ms
        f_min: int = 20,
        f_max: int | None = None,
        log: bool = True,
        snip_edges: bool = True,
    ):
        super().__init__()

        # ── basic attributes ───────────────────────────────────────────────────
        self.sr              = int(sr)
        self.n_mels          = int(n_mels)
        self.target_time_bins = int(target_time_bins)
        self.clip_length     = float(clip_length)          # seconds
        self.ratio           = float(ratio)
        self.snip_edges      = snip_edges
        self.log             = log

        self.f_min = int(f_min)
        self.f_max = int(sr // 2 if f_max is None else f_max)

        # ── decide hop & frame length (in *ms*) ───────────────────────────────
        if hop_length is None or frame_length is None:
            num_samples = int(round(self.clip_length * self.sr))
            frame_ms, hop_ms = hop_and_frame_for_target(
                num_samples,
                self.sr,
                self.target_time_bins,
                ratio=self.ratio,
                snip_edges=self.snip_edges,
            )
            # allow explicit user overrides
            self.hop_length   = hop_ms   if hop_length   is None else hop_length
            self.frame_length = frame_ms if frame_length is None else frame_length
        else:
            self.hop_length   = float(hop_length)
            self.frame_length = float(frame_length)

    # ───────────────────────────────────────────────────────────────────────────
    # Forward
    # ───────────────────────────────────────────────────────────────────────────
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : (channels, samples) or (samples) tensor
            Input audio.  Stereo is down-mixed to mono.

        Returns
        -------
        spec : (1, target_time_bins, n_mels) tensor
        """

        # ── zero-center the waveform ──────────────────────────────────────────            
        waveform = waveform - waveform.mean()

        # ── log-mel FBanks via Kaldi wrapper ──────────────────────────────────
        spec = fbank(
            waveform,
            sample_frequency=self.sr,
            frame_length=self.frame_length,       # ms
            frame_shift=self.hop_length,          # ms
            num_mel_bins=self.n_mels,
            high_freq=self.f_max,
            low_freq=self.f_min,
            use_log_fbank=self.log,
            window_type="hanning",
            snip_edges=self.snip_edges,
        )  # (time, n_mels)

        # ── pad / truncate to exactly target_time_bins ────────────────────────
        T = spec.shape[0]
        if T < self.target_time_bins:                         # pad at end
            pad = self.target_time_bins - T
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad))
        elif T > self.target_time_bins:                       # truncate
            spec = spec[: self.target_time_bins]

        # final shape: (1, time, mel)
        return spec.unsqueeze(0)
    
    
