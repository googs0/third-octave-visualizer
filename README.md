### Groups FFT bins into IEC-style 1/3-octave bands (25 Hz to 16 kHz), with adjustable A-weighting and peak-hold caps.

## Features
- 29 standard 1/3-octave bands
- Toggle A-weighting on band powers
- Adjustable smoothing & visual gain
- Peak-hold caps for that vintage analyzer look

## Controls

`A` — toggle A-weighting on/off

`←` / `→` — increase/decrease smoothing (EMA)

`↑` / `↓` — increase/decrease visual gain

`B` — toggle peak-hold caps

`Q` / `ESC` — quit

## Notes
- **Display levels** are relative for visual comparison, not absolute SPL
- **Bands** are computed by summing FFT power between band edges defined by `2^(±1/6)` from center frequencies
- **Uses a Hann window**; frame length defaults to NFFT=4096

## Troubleshooting
- Frozen bars / zero: Mic permissions or wrong device—set `DEVICE` in code or allow mic access
- Too jumpy: Increase smoothing (`←`) or make **NFFT** larger
- Too flat / clipped visually: Adjust gain (`↑`/`↓`)

## Keywords
**FFT** = Fast Fourier Transform takes signal over a period of time and breaks it up into singular frequency components \
**IEC** = International Electricotechnical Commission \
**Plain wording:** Engineers use an FFT algorithm to meet various IEC standards 
