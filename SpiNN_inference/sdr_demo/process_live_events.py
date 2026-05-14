"""
This entry script handles the streaming of live events acquired from DAVIS346 DVS to the SpiNNaker
hardware. It also manages the synchronization between sending spikes through Injector
and collecting classification results with Receiver classes.
"""

import numpy as np
from configs import InjectorConfig, ReceiverConfig
from digit_recognition.SpiNN_inference.helpers import receiver_nmnist, injector

delta_t = 1_000 # Event collection window in microseconds

inject_cfg = InjectorConfig()
receiver_cfg = ReceiverConfig()

# Initialize SpiNNaker communication interfaces
spinn_injector = injector.Injector(Injector_conf=inject_cfg)
spinn_receiver = receiver_nmnist.Receiver(receiver_conf=receiver_cfg)
spinn_injector.database_ready.wait()

# Iterator which yields events sliced by delta_t
MV_EventsIterator = spinn_injector.DvEventsIterator(
    input_path=inject_cfg.INPUT_PATH,  delta_t=delta_t)

print("After DvEventsIterator")
cnt = 0

n_min_active = (32*32*1)*0.02 # minimum active events threshold
open(receiver_cfg.FILE_NAME_PREDICTIONS, "w").close()
open(receiver_cfg.FILE_NAME_SPIKE_COUNTS, "w").close()

for ev in MV_EventsIterator:
        if len(ev) > 0:
            ev = ev[ev['p'] == 1]
        if len(ev) > n_min_active:

            spinn_injector.inject_batch_events(ev)

            # Signal the receiver to finalize a prediction window
            if (cnt%(100_000/delta_t)) == 0:
                flag_ending_packet = True
            else:
                flag_ending_packet = False
            # Retrieve spike counts from the output layer
            wta = spinn_receiver.process_output_events(
                cnt, flag_ending_packet = flag_ending_packet)
            cnt += 1
            if wta is not None and len(wta) > 0:
                # Perform classification
                prediction = np.argmax(wta)