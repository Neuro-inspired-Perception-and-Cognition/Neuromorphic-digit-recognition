"""
This class defines the parameters for neuron populations,
such as membrane time constants, capacitance, and threshold potentials.
"""

class CellParams:
    def __init__(self):

        self.gaussian_neuron_parameters = {
            'tau_m': 120.0,
            'cm': 1.0,
            'v_rest': -65.0,
            'v_reset': -65.0,
            'v_thresh': 0.0,
            'tau_syn_E': 10.0,
            'tau_syn_I': 10.0,
            'tau_refrac': 0.1,
            'i_offset': 0.0}

        self.excitatory_neuron_L2 = self.excitatory_neuron_L1 = {
            'tau_m': 90.0,
            'cm': 1.0,
            'v_rest': -65.0,
            'v_reset': -65.0,
            'v_thresh': -50.0,
            'tau_syn_E': 10.0,
            'tau_syn_I': 10.0,
            'tau_refrac': 5.0,
            'i_offset': 0.0
        }

    def get(self, name):
        return getattr(self, name, None)