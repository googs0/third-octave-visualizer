# 1/3-Octave Equalizer Visualizer
# Real-time spectrum bars grouped into standard 1/3-octave bands
# Toggle A-weighting to see how the spectrum shifts and reshapes

# Controls:
#   A        → toggle A-weighting on/off (applies in the frequency domain to band powers)
#   ← / →    → change smoothing (decay)
#   ↑ / ↓    → change gain (visual scaling)
#   B        → toggle peak-hold caps on bars
#   Q or ESC → quit

import numpy as np
import sounddevice as sd
import pygame


FS = 48000
NFFT = 4096                 # FFT size per frame
HOP = NFFT                  # simple frame-by-frame (can reduce for faster refresh)
DEVICE = None               # pick a specific mic if needed

APPLY_A_WEIGHT = False
SMOOTH = 0.6                # EMA smoothing for bar values (0 - 1), bigger = smoother
GAIN = 1.0                  # visual scale multiplier
USE_PEAK_HOLD = True
PEAK_DECAY = 0.98           # how fast the peak caps fall (0 - 1), closer to 1 = slower decay

# Standard 1/3-octave band centers (IEC 61260) from 25 Hz to 16 kHz
CENTER_HZ = np.array([
     25,   31.5,   40,   50,   63,   80,  100,  125,  160,  200,
    250,   315,   400,  500,  630,  800, 1000, 1250, 1600, 2000,
   2500,  3150,  4000, 5000, 6300, 8000, 10000, 12500, 16000
], dtype=float)

def band_edges_from_centers(fc):
    """Compute 1/3-octave band edges from band centers.
    For 1/3-octave: factor = 2**(1/6) between center and edges.
    """
    k = 2 ** (1.0 / 6.0)
    f1 = fc / k  # lower edge
    f2 = fc * k  # upper edge
    return f1, f2

def a_weight_db(f):
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 1e-6)
    ra = (12194**2 * f**4) / (
        (f**2 + 20.6**2)
        * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
        * (f**2 + 12194**2)
    )
    return 20.0 * np.log10(ra) + 2.0

def compute_band_powers(samples, fs, centers, apply_a=False):
    """Return power per 1/3-octave band for the given samples.
    Uses FFT power and sums bins within each band's [f1,f2].
    If apply_a=True, multiply each bin's power by A-weighting factor (linear).
    """
    # Window
    win = np.hanning(len(samples))
    xw = samples * win

    X = np.fft.rfft(xw, n=len(samples))
    P = (np.abs(X) ** 2) / np.sum(win**2)  # relative power
    freqs = np.fft.rfftfreq(len(samples), d=1.0/fs)

    if apply_a:
        A_lin = 10.0 ** (a_weight_db(freqs) / 10.0)
        P = P * A_lin

    # Sum bins per band
    f1, f2 = band_edges_from_centers(centers)
    band_power = np.zeros_like(centers, dtype=float)
    for i, (lo, hi) in enumerate(zip(f1, f2)):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size > 0:
            band_power[i] = P[idx].sum()
        else:
            band_power[i] = 0.0
    return band_power

def main():
    global APPLY_A_WEIGHT, SMOOTH, GAIN, USE_PEAK_HOLD

    pygame.init()
    w, h = 1000, 400
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("1/3-Octave Equalizer Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Precompute graphics layout
    n_bands = len(CENTER_HZ)
    margin = 40
    spacing = 6
    bar_w = (w - 2*margin - spacing*(n_bands-1)) // n_bands
    bar_bottom = h - margin

    # Audio stream
    frames_per_block = HOP
    stream = sd.InputStream(samplerate=FS, blocksize=frames_per_block, channels=1, dtype="float32", device=DEVICE)
    stream.start()

    # Bar state
    bar_vals = np.zeros(n_bands, dtype=float)
    bar_peaks = np.zeros(n_bands, dtype=float)

    try:
        while True:
            data, _ = stream.read(frames_per_block)
            mono = data[:, 0].astype(np.float64)

            # If NFFT > len, zero-pad for better frequency resolution consistency
            if len(mono) < NFFT:
                samples = np.zeros(NFFT, dtype=np.float64)
                samples[:len(mono)] = mono
            else:
                samples = mono[:NFFT]

            band_power = compute_band_powers(samples, FS, CENTER_HZ, apply_a=APPLY_A_WEIGHT)

            # Convert power to dB for display (relative); add tiny epsilon to avoid log(0).
            band_db = 10.0 * np.log10(band_power + 1e-20)

            # Normalize to 0 - 1 via min/max over range
            # Linearly map [-80 dB to 0 dB] to [0 - 1] then apply gain.
            norm = np.clip((band_db + 80.0) / 80.0, 0.0, 1.0) * GAIN
           
          # Smooth (EMA)
            bar_vals = SMOOTH * bar_vals + (1.0 - SMOOTH) * norm

            # Peak hold
            if USE_PEAK_HOLD:
                bar_peaks = np.maximum(bar_peaks * PEAK_DECAY, bar_vals)

            # Draw
            screen.fill((20, 20, 24))

            # Bars
            x = margin
            for i, v in enumerate(bar_vals):
                # determine color from value (blue→green→yellow→red)
                val = float(np.clip(v, 0.0, 1.0))
                if val < 0.33:
                    t = val / 0.33
                    color = (0, int(255*t), 255)          # blue to cyan
                elif val < 0.5:
                    t = (val - 0.33) / (0.5 - 0.33)
                    color = (0, 255, int(255*(1-t)))      # cyan to green
                elif val < 0.66:
                    t = (val - 0.5) / (0.66 - 0.5)
                    color = (int(255*t), 255, 0)          # green to yellow
                else:
                    t = (val - 0.66) / (1.0 - 0.66)
                    color = (255, int(255*(1-t)), 0)      # yellow to red

                bar_h = int(val * (h - 2*margin))
                rect = pygame.Rect(x, bar_bottom - bar_h, bar_w, bar_h)
                pygame.draw.rect(screen, color, rect)

                if USE_PEAK_HOLD:
                    peak_h = int(bar_peaks[i] * (h - 2*margin))
                    cap_rect = pygame.Rect(x, bar_bottom - peak_h - 4, bar_w, 4)
                    pygame.draw.rect(screen, (240, 240, 240), cap_rect)

                # Center frequency label
                label = font.render(f"{int(CENTER_HZ[i])}", True, (200, 200, 210))
                label_rect = label.get_rect(center=(x + bar_w//2, bar_bottom + 12))
                screen.blit(label, label_rect)

                x += bar_w + spacing

            # UI txt
            info = f"A-weight: {'ON' if APPLY_A_WEIGHT else 'OFF'}  Smooth: {SMOOTH:.2f}  Gain: {GAIN:.2f}  Peaks: {'ON' if USE_PEAK_HOLD else 'OFF'}"
            screen.blit(font.render(info, True, (230, 230, 235)), (10, 8))

            pygame.display.flip()
            clock.tick(30)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_a:
                        APPLY_A_WEIGHT = not APPLY_A_WEIGHT
                    elif event.key == pygame.K_LEFT:
                        SMOOTH = min(0.95, max(0.0, SMOOTH + 0.05))
                    elif event.key == pygame.K_RIGHT:
                        SMOOTH = min(0.95, max(0.0, SMOOTH - 0.05))
                    elif event.key == pygame.K_UP:
                        GAIN = min(5.0, GAIN * 1.15)
                    elif event.key == pygame.K_DOWN:
                        GAIN = max(0.2, GAIN / 1.15)
                    elif event.key == pygame.K_b:
                        USE_PEAK_HOLD = not USE_PEAK_HOLD

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        pygame.quit()

if __name__ == "__main__":
    main()
