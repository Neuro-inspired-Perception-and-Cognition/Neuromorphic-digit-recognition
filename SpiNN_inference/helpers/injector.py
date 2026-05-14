"""
This file provides the Injector class with functions used for streaming
event data to a SpiNNaker neuromorphic board over an Ethernet live connection.
"""

import os
import cv2
import shutil
import threading

import numpy as np
import dv_processing as dv
import spynnaker.pyNN as sim

from datetime import timedelta
from tonic.io import make_structured_array
from tonic.io import read_aedat4, read_mnist_file, events_struct


class Injector:
    """
    Class to establish a live connection for sending events to a SpiNNaker board via Ethernet.

    database_ready : Event to wait until the SpiNNaker database is ready.
    DvEventsIterator(input_path, delta_t, drop_last): creates an iterator over a folder with data,
                                                      given AEDAT4 file or live camera input.
    inject_events(): sends AEDAT4 spike events using the iterator.
    inject_batch_events(events): sends a given batch of events to the SpiNNaker board.
    """

    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Return existing instance if already created
            cls._instance = super(Injector, cls).__new__(cls)
        return cls._instance
    def __init__(self, Injector_conf):
        if not hasattr(self, 'initialized'):
            self.Injector_conf = Injector_conf
            self.SENDER_PORT = self.Injector_conf.SENDER_PORT
            self.add_callback = self.Injector_conf.ADD_CALLBACK

            # Open a live spike connection to the SpiNNaker board on the configured port
            self.connection = sim.external_devices.SpynnakerLiveSpikesConnection(
                local_port=self.SENDER_PORT, send_labels=[self.Injector_conf.INJECTOR_SPINN_POP_LABEL])
            if self.add_callback == True:
                self.connection.add_start_resume_callback(self.Injector_conf.INJECTOR_SPINN_POP_LABEL, self.set_database_ready)
                print("\nInjector: Call-back added ...\n")

            self.database_ready = threading.Event()

            # Visualisation buffers
            self.frame = np.zeros(( self.Injector_conf.N_POL, self.Injector_conf.FOV_H, self.Injector_conf.FOV_W), dtype=np.uint8)
            self.play_back_frame_full = np.zeros(( self.Injector_conf.DVS_H, self.Injector_conf.DVS_W, 3), dtype=np.uint8)
            self.play_back_frame_cropped = np.zeros(( self.Injector_conf.FOV_H, self.Injector_conf.FOV_W, 3), dtype=np.uint8)
            self.img_cnt = 0 # Counter for saved frame filenames
            if self.Injector_conf.SAVE_FIG == True:
                if (os.path.exists(self.Injector_conf.SAVE_FIG_PATH)):
                    shutil.rmtree(self.Injector_conf.SAVE_FIG_PATH)
                os.mkdir(self.Injector_conf.SAVE_FIG_PATH)

            # Background activity noise filter to suppress noise events from live camera input
            resolution = (self.Injector_conf.DVS_W, self.Injector_conf.DVS_H)
            self.filter = dv.noise.BackgroundActivityNoiseFilter(resolution, backgroundActivityDuration=timedelta(milliseconds=5))


    def DvEventsIterator(self, input_path, delta_t, drop_last = True):
        """
        Yield succesive event batches of delta_t microseconds from one of three sources:
          1. A directory of .bin files
          2. A single AEDAT4 recording file
          3. A live DAVIS camera stream
        """
        if os.path.isdir(input_path):
            # Streaming input dataset
            for root, _, files in os.walk(input_path):
                for fname in files:
                    if not fname.endswith(".bin"):
                        continue

                    file_path = os.path.join(root, fname)
                    evts = read_mnist_file(file_path, dtype=events_struct)

                    if len(evts) == 0:
                        continue

                    evts.sort(order="t")

                    recording_duration = evts["t"][-1] - evts["t"][0]
                    n_iter, last_packet = divmod(recording_duration, delta_t)
                    n_iter = int(n_iter)
                    start_ts = evts["t"][0]

                    for i in range(n_iter):
                        yield evts[
                            ((start_ts + delta_t * i) <= evts["t"]) &
                            (evts["t"] < (start_ts + delta_t * (i + 1)))
                            ]

                    if (last_packet > 0) and (drop_last is False):
                        yield evts[(start_ts + delta_t * n_iter) <= evts["t"]]
        elif input_path != "":
            # Streaming single input aedat4 file
            evts = read_aedat4(input_path)
            recording_duration = evts["t"][-1] - evts["t"][0]
            n_iter, last_packet = divmod( recording_duration , delta_t)
            n_iter = int(n_iter)
            start_ts = evts["t"][0]
            for i in range(n_iter):
                yield evts[ ( (start_ts + delta_t*i) <= evts["t"]) & (evts["t"] < (start_ts + delta_t*(i + 1)) ) ]

            if (last_packet>0) and (drop_last == False):
                yield evts[ (start_ts + delta_t*(n_iter))<=evts["t"] ]
        else:
            # Streaming live camera input
            print(f"\n{'=' * 60}")
            print(f"Expected resolution: {self.Injector_conf.DVS_W}×{self.Injector_conf.DVS_H}")
            print(f"{'=' * 60}\n")

            capture = dv.io.camera.open()
            capture.setTimeInterval(timedelta(microseconds=delta_t))

            # flush old events
            while capture.getNextEventBatch() is not None:
                pass

            slicer = dv.EventStreamSlicer()
            accumulated = []

            def on_slice(events):
                # filter incoming events from the camera
                self.filter.accept(events)
                filtered = self.filter.generateEvents()
                # accumulate events into delta_t windows
                accumulated.append(filtered.numpy())

            slicer.doEveryTimeInterval(timedelta(microseconds=delta_t), on_slice)

            while capture.isRunning():
                batch = capture.getNextEventBatch()
                if batch is not None:
                    slicer.accept(batch)
                if accumulated:
                    if len(accumulated) > 1:
                        print(len(accumulated))
                    evts = accumulated.pop(0)
                    accumulated.clear()
                    xytp = make_structured_array(
                        evts['y'], evts['x'], evts['timestamp'], evts['polarity'])
                    yield xytp


    def set_database_ready(self, pop_lable, events):
        print("\nDatabase is ready\n")
        self.database_ready.set()

    def inject_batch_events(self, events):
        """
        Prepare event batch for sending to SpiNNaker
        with visualization of received events
        """

        polarity = "p"
        min_number_of_events = self.Injector_conf.MIN_EVT_TO_SAVE
        if (len(events) == 0):
            pass
        else:
            if self.Injector_conf.FLIP_INPUT == True:
                events["y"] = self.Injector_conf.DVS_H-1 - events["y"]

            self.play_back_frame_full = np.zeros(( self.Injector_conf.DVS_H, self.Injector_conf.DVS_W, 3), dtype=np.uint8)
            self.play_back_frame_cropped = np.zeros(( self.Injector_conf.FOV_H, self.Injector_conf.FOV_W, 3), dtype=np.uint8)

            if len(events)>0:
                self.play_back_frame_full[ events[events[polarity]==0]['x'], events[events[polarity]==0]['y'], 0] = 255
                self.play_back_frame_full[ events[events[polarity]==1]['x'], events[events[polarity]==1]['y'], 1] = 255

                if self.Injector_conf.N_POL == 1:
                    events = events[events[polarity]==1] # Only positive polarity
                else:
                    self.frame[events[polarity].astype("int8"), events['x'], events['y']] = 255

                events = events[(events['x']<self.Injector_conf.FOV_W)&(events['y']<self.Injector_conf.FOV_H)]
                self.frame[0, events['x'], events['y']] = 255
                self.play_back_frame_cropped[ events[events[polarity]==0]['x'], events[events[polarity]==0]['y'], 0] = 255
                self.play_back_frame_cropped[ events[events[polarity]==1]['x'], events[events[polarity]==1]['y'], 1] = 255

                n_id = np.where(self.frame.flatten()!=0)[0]
                self.connection.send_spikes(self.Injector_conf.INJECTOR_SPINN_POP_LABEL, n_id)

                self.frame = np.zeros(( self.Injector_conf.N_POL, self.Injector_conf.FOV_H, self.Injector_conf.FOV_W), dtype=np.uint8)

                cv2.imshow('Camera: Full', self.play_back_frame_full)
                cv2.imshow('Camera: Cropped', cv2.resize(self.play_back_frame_cropped,  (self.Injector_conf.FOV_W*3, self.Injector_conf.FOV_H*3)))
                cv2.waitKey(1)

                if self.Injector_conf.SAVE_FIG:
                    if len(events)>min_number_of_events:
                        cv2.imwrite(f"{self.Injector_conf.SAVE_FIG_PATH}/{self.img_cnt}.png", self.play_back_frame_cropped )#, cmap=plt.cm.hot)
                self.img_cnt+=1