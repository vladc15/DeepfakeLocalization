# Deepfake Localization Across Generative Models Using Deep Learning

This repository contains the code for my bachelor's thesis, which explores deepfake localization across different generative domains, focusing on generalization and proposing approaches to enhance robustness.

## Requirements

The project requires extensive GPU resources. The code has been designed to run on Google Colab, but can be adapted to local environments.

Implementation is based on the [DeCLIP](https://github.com/bit-ml/DeCLIP) architecture and methods presented in the [paper](https://arxiv.org/abs/2409.08849). An additional file `declip_utils.py` is required, containing the utilitary functions from `utils.py` of the DeCLIP repository.

The used dataset in our study is the [Dolos](https://github.com/bit-ml/dolos) dataset, suitable for deepfake localization tasks, as explained in the [paper](https://arxiv.org/abs/2311.04584). It should be employed in the recommended format, downloaded and placed under the `./datasets/` root directory.

## References

- https://github.com/bit-ml/DeCLIP
- https://github.com/bit-ml/dolos

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">