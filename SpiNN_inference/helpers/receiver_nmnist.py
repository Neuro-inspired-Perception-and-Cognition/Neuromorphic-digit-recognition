"""
This file provides the Receiver class for collecting spike outputs from a
SpiNNaker neuromorphic board over an Ethernet live connection.
"""

import cv2, time
import warnings
import numpy as np
import spynnaker.pyNN as sim
warnings.filterwarnings('ignore')

class Receiver:
    """
    Class for receiving events from a SpiNNaker board via Ethernet.
    Monitors three SpiNNaker populations (downScaledPop, pop_hidden_1, pop_out)
    and accumulates incoming spikes to produce classification predictions.
    """
    _instance =None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Receiver, cls).__new__(cls)
        return cls._instance
    def __init__(self, receiver_conf):
        if not hasattr(self, 'initialized'):

            self.receiver_conf = receiver_conf

            self.W, self.H = 32, 32
            self.wta_out = np.zeros((10))
            self.argmaxs_array = []
            self.wta_out_accumulated = np.zeros((10))
            self.temp_maps_downScaled = np.zeros((self.W, self.H, 3), dtype=np.uint8)

            self.receive_labels = ["downScaledPop", "pop_hidden_1", "pop_out"]
            self.connection = sim.external_devices.SpynnakerLiveSpikesConnection(
                    local_port=self.receiver_conf.RECEIVER_PORT, receive_labels=self.receive_labels)#,send_labels=None)

            if self.receiver_conf.ADD_CALLBACK == True:
                for pop_filter_neurons_id in range(len(self.receive_labels)):
                    self.connection.add_receive_callback(self.receive_labels[pop_filter_neurons_id], self.receive_spikes)
                print("\nReceiver: Call-back added ...\n")

            print("Waiting ...")

            self.n_ids = np.empty(0)
            self.data_pos_in_pop_out = []
            self.all_pop_pos_data = []
            self.detections = []
            self.num_all_spikes_for_energy = {label: 0 for label in self.receive_labels}


    def receive_spikes(self, label, time, neuron_ids):
        """
        registered callback from SpiNNaker,
        counts incoming spikes to the buffer for prediction and energy estimation
        """
        if True:
            pos_data = np.array(neuron_ids).astype(np.uint32)
            self.all_pop_pos_data.append({'neurons_id':pos_data, 'pop_label':label})
            self.num_all_spikes_for_energy[label] += len(pos_data)
        return


    def show_bar_graph(self, values,
                    win_name="Bar Graph",
                    size=(500, 300),
                    max_val=100):
        """
        Plots a bar graph of the ten output neuron spike counts
        and highlights corresponding prediction in red
        """
        assert len(values) == 10, "values must have length 10"
        W, H = size
        N = 10

        BG_COLOR = (30, 30, 30)
        BAR_COLOR = (0, 255, 0)
        BAR_COLOR_MAX = (0, 0, 255)
        TEXT_COLOR = (255, 255, 255)

        bar_width = W // N
        img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        v_max = np.max(values)
        for i, val in enumerate(values):
            val = max(0, min(val, max_val))  # clamp
            bar_h = int((val / max_val) * (H - 40))

            x1 = i * bar_width + 5
            y1 = H - 20
            x2 = x1 + bar_width - 10
            y2 = y1 - bar_h
            if val == v_max:
                cv2.rectangle(img, (x1, y1), (x2, y2), BAR_COLOR_MAX, -1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), BAR_COLOR, -1)

            cv2.putText(img, str(i), (x1, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 1)
            cv2.putText(img, str(int(val)), (x1, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

        cv2.imshow(win_name, img)

    def neuron_id_to_downScaled_map(self, neurons_id_array, out_size_y, out_size_x):
        """
        Converts a flat array of neuron ids back into a spatial activity map.
        """
        mapped = np.zeros((out_size_y, out_size_x, 3), dtype=np.uint8)
        sum_X, sum_Y = 0, 0
        for neurons_id in neurons_id_array:
            P = neurons_id//(out_size_x*out_size_y)
            neurons_id-=P*out_size_x*out_size_y

            Y = neurons_id//out_size_x
            X = neurons_id % out_size_x
            mapped[Y, X, 2] = 255
            sum_X += X
            sum_Y += Y
        return mapped


    def extract_recordings_into_maps(self, pop_pos_data):
        """
        Routes a spike batch to the appropriate accumulator based on its source population.
        """
        neurons_id = pop_pos_data['neurons_id']
        if pop_pos_data['pop_label'] == 'downScaledPop':
            # reconstruct and accumulate spatial activity map
            out_size_y, out_size_x = (self.W, self.H)
            self.temp_maps_downScaled+= self.neuron_id_to_downScaled_map(neurons_id, out_size_y, out_size_x)

        if pop_pos_data['pop_label'] == 'pop_out':
            # increment the per-class spike counter used for classification
            for n_id in neurons_id:
                self.wta_out[n_id] += 1

    def show_gaussian(self):
        """
        Displays the accumulated downscaled population activity map
        """
        cv2.imshow(f'Gaussian Population from SpiNNaker', cv2.resize(self.temp_maps_downScaled, (200, 200)))
        cv2.waitKey(1)
        self.temp_maps_downScaled = np.zeros((self.W, self.H, 3), dtype=np.uint8)

    def process_output_events(self, time_tick_cnt, flag_ending_packet = False):
        """
        Processes all buffered spikes for the current time window, accumulates
        output population activity, and makes a classification prediction when an
        ending packet is received.
        """
        self.wta_out = np.zeros((10))

        for pop_pos_data in self.all_pop_pos_data:
            self.extract_recordings_into_maps( pop_pos_data)
        self.data_pos_in_pop_out = []
        self.all_pop_pos_data = []

        if flag_ending_packet == True:
            if self.wta_out_accumulated.sum()>0:
                norm_wta_out = (self.wta_out_accumulated - self.wta_out_accumulated.min()) / (self.wta_out_accumulated.max() - self.wta_out_accumulated.min())
                soft_max_like_P = norm_wta_out/sum(norm_wta_out)
                pred_time = time.time()
                pred = np.argmax(self.wta_out_accumulated)
                bar = soft_max_like_P*100
                print(f"Prediction: {np.argmax(self.wta_out_accumulated)} at {time.time()}")

                self.show_bar_graph(
                    values=self.wta_out_accumulated,
                    max_val=max(self.wta_out_accumulated) if self.wta_out_accumulated.max() > 0 else 1
                )
                self.detections.append((np.argmax(self.wta_out_accumulated), time.time()))
                with open(self.receiver_conf.FILE_NAME_PREDICTIONS, "a") as f:
                    f.write(f"{pred},{pred_time},{bar.tolist()}\n")
                with open(self.receiver_conf.FILE_NAME_SPIKE_COUNTS, "a") as f:
                    f.write(f"{self.num_all_spikes_for_energy},{pred_time}\n")
                self.show_gaussian()
                self.wta_out_accumulated = np.zeros((10))
        else:
            self.wta_out_accumulated += self.wta_out

        if time_tick_cnt%100 == 0:
            if len(self.argmaxs_array)>0:
                self.argmaxs_array = []
        else:
            self.argmaxs_array.append(np.argmax(self.wta_out_accumulated))

        return self.wta_out