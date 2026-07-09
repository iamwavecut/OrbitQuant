# OrbitQuant 0.1.4 Release Notes

OrbitQuant 0.1.4 is a release-metrics patch for external GenEval and VBench
metric import.

## Install

```bash
pip install "orbitquant[hf]==0.1.4"
```

For optimized packed-weight inference dependencies:

```bash
pip install "orbitquant[kernels]==0.1.4"
```

## Changes

- Imported GenEval `geneval_overall` now follows upstream GenEval semantics:
  average over task scores.
- GenEval image-level and prompt-level hit rates are imported as
  `geneval_image_accuracy` and `geneval_prompt_accuracy` when they are present
  in the external results.
- VBench external eval commands now pass requested dimensions as separate CLI
  arguments, matching upstream `--dimension` parsing.
- VBench prompt files now use exported video filenames as keys for custom input
  evaluation.
- README and generated model cards document GenEval and VBench release metric
  semantics.

## Claim Boundary

This release does not change quantization math, artifact format, runtime
dispatch, kernel implementations, or published model weights. Release-grade
GenEval/VBench runs remain separate from this package patch.
