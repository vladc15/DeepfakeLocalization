# Deepfake Localization Across Generative Models Using Deep Learning

This repository contains the code for my bachelor's thesis, which explores deepfake localization across different generative domains, focusing on generalization and proposing approaches to enhance robustness.

## Requirements

The project requires extensive GPU resources. The code has been designed to run on Google Colab, but can be adapted to local environments.

Implementation is based on the [DeCLIP](https://github.com/bit-ml/DeCLIP) architecture and methods presented in the [paper](https://arxiv.org/abs/2409.08849). An additional file `declip_utils.py` is required, containing the utilitary functions from `utils.py` of the DeCLIP repository.

The used dataset in our study is the [Dolos](https://github.com/bit-ml/dolos) dataset, suitable for deepfake localization tasks, as explained in the [paper](https://arxiv.org/abs/2311.04584). It should be employed in the recommended format, downloaded and placed under the `./datasets/` root directory.

## References

- https://github.com/bit-ml/DeCLIP
- https://github.com/bit-ml/dolos