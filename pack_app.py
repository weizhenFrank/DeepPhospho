"""
Packing DeepPhospho APP with four types:
    - Only source code
    - Source code with pre-trained model parameters
    - Source code with required python environment
    - Source code with required python environment and pre-trained model parameters
"""

from sys import argv
import os

if __name__ == '__main__':
    if len(argv) == 2:
        conda_env_folder = argv[1]
    else:
        conda_env_folder = r''

    if not isinstance(conda_env_folder, str):
        raise ValueError
    if not os.path.exists(conda_env_folder):
        raise FileNotFoundError
    if not os.path.isdir(conda_env_folder):
        raise ValueError

    # TODO
