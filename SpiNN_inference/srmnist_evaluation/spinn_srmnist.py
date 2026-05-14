"""
This entry script builds a spiking neural network sRMNIST model for deploying
on SpiNNaker neuromorphic platform and starts endless inference loop
"""
import spynnaker.pyNN as p
from configs import ModelConfig
from digit_recognition.SpiNN_inference.helpers.cell_params import CellParams
from digit_recognition.SpiNN_inference.helpers.SpiNNaker_helpers import *

def load_connections(file_path, w_scale = 5, make_inh_positive=True):# 100
    """
    Reads a text file with columns i j weight delay
    Returns
        exc_connections: [(i, j, w, d)] where w > 0
        inh_connections: [(i, j, abs(w), d)] where w < 0
    """
    exc_connections = []
    inh_connections = []

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid line at {line_num} (expected 4 columns): {line}"
                )

            pre = int(parts[0])
            post = int(parts[1])
            weight = float(parts[2])*w_scale
            delay = float(parts[3])

            if weight > 0:
                exc_connections.append((pre, post, weight, delay))
            elif weight < 0:
                w = abs(weight) if make_inh_positive else weight
                inh_connections.append((pre, post, w, delay))
            # weight == 0 ignored

    return exc_connections, inh_connections

def print_layer_sizes(input_pop, pop_hidden_1, pop_out):
    print("\n===== NEURON COUNTS =====")
    print(f"conv1          : {input_pop.size}")
    print(f"fc1            : {pop_hidden_1.size}")
    print(f"output_pop     : {pop_out.size}")
    print("=========================\n")
    print(f"TOTAL neurons  : {input_pop.size + pop_hidden_1.size + pop_out.size}")
    print("=========================\n")

model_cfg = ModelConfig()
params = CellParams()
excitatory_neuron_L1 = params.excitatory_neuron_L1
excitatory_neuron_L2 = params.excitatory_neuron_L2

p.setup(timestep=1)
input_size = (1, 32, 32)
# p.set_number_of_neurons_per_core(p.IF_curr_exp, 15)

# Live spike injector = 32x32 input to sRMNIST
input_label = 'pop_inp'
input_pop = p.Population(
    np.prod(input_size),
    p.external_devices.SpikeInjector(
        database_notify_port_num=model_cfg.SENDER_PORT
    ),
    label=input_label
)

# Hidden layer (L2)
pop_hidden_1_label = 'pop_hidden_1'
pop_hidden_1 = p.Population(100,
                   p.IF_curr_exp(**excitatory_neuron_L1),
                   label = pop_hidden_1_label)
exc, inh = load_connections(r"C:\Users\vedme\OneDrive\Робочий стіл\BT\code\digit_recognition\SpiNN_inference\model_weights\ConnectionFile_w_fc1_torch.Size([1024, 100])_SRMNIST.txt")
print(f"downscale -> hidden (exc)   : {len(exc)}")
print(f"downscale -> hidden (inh)   : {len(inh)}")
downScaled_pop_hidden_1_exc_prj = p.Projection(input_pop, pop_hidden_1, p.FromListConnector(exc), receptor_type='excitatory')
downScaled_pop_hidden_1_inh_prj = p.Projection(input_pop, pop_hidden_1, p.FromListConnector(inh), receptor_type='inhibitory')

# Output layer (L3)
pop_out_label = 'pop_out'
pop_out = p.Population(10,
                   p.IF_curr_exp(**excitatory_neuron_L2),
                   label = pop_out_label)
exc, inh = load_connections(r"C:\Users\vedme\OneDrive\Робочий стіл\BT\code\digit_recognition\SpiNN_inference\model_weights\ConnectionFile_w_fc2_torch.Size([100, 10])_SRMNIST.txt")
print(f"hidden -> output (exc)      : {len(exc)}")
print(f"hidden -> output (inh)      : {len(inh)}")
pop_hidden_1_out_exc_prj = p.Projection(pop_hidden_1, pop_out, p.FromListConnector(exc), receptor_type='excitatory')
pop_hidden_1_out_inh_prj = p.Projection(pop_hidden_1, pop_out, p.FromListConnector(inh), receptor_type='inhibitory')

receive_labels = list([input_label, pop_hidden_1_label, pop_out_label])


# Winner-Take-All mechanism for L3
conn_list = []
w_inhib = 10#.5
N = pop_out.size
for i in range(N):
    for j in range(N):
        if i != j:
            conn_list.append((i, j,  w_inhib, 1))
p.Projection(
    pop_out,
    pop_out,
    p.FromListConnector(conn_list),
    receptor_type='inhibitory'
)
print(f"WTA                         : {len(conn_list)}")

RIGHT_RECEIVER_PORT = model_cfg.RIGHT_RECEIVER_PORT
READOUT_ANGLE_PORT = model_cfg.RIGHT_RECEIVER_PORT

# Live output activation to monitor spike counts
p.external_devices.activate_live_output_for(input_pop, database_notify_port_num=RIGHT_RECEIVER_PORT)
p.external_devices.activate_live_output_for(pop_hidden_1, database_notify_port_num=RIGHT_RECEIVER_PORT)
p.external_devices.activate_live_output_for(pop_out, database_notify_port_num=RIGHT_RECEIVER_PORT)

print_layer_sizes(input_pop, pop_hidden_1, pop_out)
p.external_devices.run_forever()