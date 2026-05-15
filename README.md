# Triplane Tokenization Prototype

This project is a compact PyTorch starter for the architecture in the reference image:

1. Encode an input image into a latent feature.
2. Decode that latent into three feature planes: `xy`, `xz`, and `yz`.
3. Patchify the planes and project each patch through an MLP.
4. Emit a token sequence that can be consumed by a transformer or VLM.

The file `triplane` is executable Python and includes:

- `ImageEncoder`: a small CNN stand-in for a stronger ViT/ResNet encoder.
- `TriplaneGenerator`: creates learned `xy`, `xz`, and `yz` planes.
- `PatchTokenizer`: converts plane patches into tokens.
- `sample_triplanes`: samples 3D points from the triplanes for future volumetric rendering losses.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

Run with a random input image:

```bash
python triplane
```

Run with your reference image:

```bash
python triplane --image /Users/micwilliams/Desktop/112.png
```

Expected output is shape-oriented, for example:

```text
image:   (1, 3, 128, 128)
xy/xz/yz planes: (1, 32, 64, 64) each
tokens:  (1, 192, 256)
samples: (1, 1024, 96)
```
