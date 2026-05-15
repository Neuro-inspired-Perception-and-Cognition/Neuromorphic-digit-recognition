"""
Configuration classes for SpiNNaker communication and modeling.

InjectorConfig
    Configuration for sending events to SpiNNaker via Ethernet.

ReceiverConfig
    Configuration for receiving events from SpiNNaker via Ethernet.

ModelConfig
    Configuration for setting parameters of the PyNN model.
"""

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectorConfig:
    # Prophesee EVK4: 1280×720
    # DAVIS346: 346×260
    FOV_W: int = 32
    FOV_H: int = 32
    DVS_W: int = 346
    DVS_H: int = 260

    N_POL: int = 1
    SENDER_PORT: int = 56786
    TIME_WINDOW: int = 10_000
    INJECTOR_SPINN_POP_LABEL = "pop_inp"
    INPUT_PATH = r'data\N-MNIST\Test' # Path to N-MNIST folder
    SAVE_FIG: bool = False
    MIN_EVT_TO_SAVE: int = -1
    OUTPUT_DIR: str = "figures/"
    SAVE_FIG_PATH: str = f"{OUTPUT_DIR}MV-Recording-{time.time()}_T-{TIME_WINDOW}_Injector"
    FLIP_INPUT: bool = False
    ADD_CALLBACK: bool = True

@dataclass(frozen=True)
class ModelConfig:
    RIGHT_RECEIVER_PORT: int = 56789
    SENDER_PORT: int = 56786

@dataclass(frozen=True)
class ReceiverConfig:
    FOV_W: int = 32
    FOV_H: int = 32
    N_POL: int = 1
    RECEIVER_PORT: int = 56789
    ADD_CALLBACK: bool = True
    FILE_NAME_PREDICTIONS = r"experiments\detections.txt"
    FILE_NAME_SPIKE_COUNTS = r"experiments\spike_counts.txt"
