# OrbitQuant

Pre-release clean-room implementation of OrbitQuant for diffusion transformer
post-training quantization.

OrbitQuant is based on the paper
[OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers](https://arxiv.org/abs/2607.02461).
The package is intended to provide Hugging Face Diffusers/Transformers adapters,
compact quantized artifacts, and native-resolution evaluation scripts.

This repository is currently private and experimental. Do not treat the current
runtime as optimized low-bit inference until the CUDA/MPS kernel work lands.

## Initial Scope

- Calibration-free RPBH + Lloyd-Max weight/activation quantization.
- Transformer-only quantization for diffusion pipelines.
- BF16 text encoders, VAE, embeddings, timestep MLP, and final heads.
- AdaLN modulation projections as INT4 RTN weight-only by default.
- Native eval settings for FLUX.2 Klein, FLUX.1-schnell, Z-Image-Turbo, and
  Wan2.1-T2V-1.3B.

## License

The code in this repository is Apache-2.0. Quantized model artifacts must record
and respect the license of their source model.
