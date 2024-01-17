import pennylane as qml
from jax import numpy as np
from pennylane.operation import Operation
from jax import jit


def u4(params, wires):
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


class U4Gate(Operation):

    nparams = 15
    num_params = 1
    num_wires = 2
    grad_method = 'A'
    name = 'U(4)'

    @staticmethod
    def compute_decomposition(*params, wires=None):
        return [u4(params[0], wires=wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "U(4) gate"


def ngate(params, wires):
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.Hadamard(wires=wires[2])
    qml.CNOT(wires=wires[1::-1])
    qml.CNOT(wires=wires[1:])
    qml.RY(2*params[0], wires=wires[1])
    qml.CNOT(wires=wires[:2])
    qml.RY(-2*params[1], wires=wires[1])
    qml.CNOT(wires=wires[:2])
    qml.CNOT(wires=wires[1:])
    qml.Hadamard(wires=wires[2])
    qml.RZ(np.pi/2, wires=wires[1])
    qml.CNOT(wires=wires[1::-1])
    qml.CNOT(wires=wires[::2])
    qml.CNOT(wires=wires[1:])
    qml.RZ(2*params[2], wires=wires[2])
    qml.CNOT(wires=wires[1:])
    qml.CNOT(wires=wires[::2])


def mgate(params, wires):
    qml.CNOT(wires=wires[2::-2])
    qml.CNOT(wires=wires[:2])
    qml.CNOT(wires=wires[2:0:-1])
    qml.RZ(-np.pi/2, wires=wires[2])
    qml.RY(2*params[0], wires=wires[2])
    qml.CNOT(wires=wires[1:])
    qml.RY(-2*params[1], wires=wires[2])
    qml.CNOT(wires=wires[1:])
    qml.RZ(np.pi/2, wires=wires[2])
    qml.CNOT(wires=wires[2::-2])
    qml.CNOT(wires=wires[:2])
    qml.CNOT(wires=wires[1::-1])
    qml.Hadamard(wires=wires[2])
    qml.CNOT(wires=wires[::2])
    qml.RZ(2*params[2], wires=wires[2])
    qml.CNOT(wires=wires[::2])
    qml.RZ(2 * params[3], wires=wires[2])
    qml.CNOT(wires=wires[1::-1])
    qml.Hadamard(wires=wires[2])


def threequbit(params, wires):
    u4(params[:15], wires=wires[:2])
    qml.U3(params[15], params[16], params[17], wires=wires[2])
    ngate(params[18:21], wires=wires)
    u4(params[21:36], wires=wires[:2])
    qml.U3(params[36], params[37], params[38], wires=wires[2])
    mgate(params[39:43], wires=wires)
    u4(params[43:58], wires=wires[:2])
    qml.U3(params[58], params[59], params[60], wires=wires[2])
    ngate(params[61:64], wires=wires)
    u4(params[64:79], wires=wires[:2])
    qml.U3(params[79], params[80], params[81], wires=wires[2])


class U8(Operation):
    nparams = 82  # number of actual params
    num_params = 1  # internal attribute, 1 since we are taking 1 array
    num_wires = 3
    grad_method = 'A'
    name = 'U(8)'

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        return [threequbit(params[0], wires=wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "U(8) gate"


class CU8(Operation):
    nparams = 82
    num_params = 1
    num_wires = qml.operation.AnyWires
    grad_method = 'A'
    name = 'CU(8)'

    def __init__(self, params, wires, ctrlstring):
        super().__init__(*params, wires=wires)
        self._hyperparameters = {"ctrlstring": ctrlstring}

    @staticmethod
    def compute_decomposition(*params, wires=None, ctrlstring=None):
        if ctrlstring is None:
            ctrlstring = '1'*(len(wires)-3)
        return [qml.ctrl(threequbit, control=wires[3:], control_values=[bool(int(i)) for i in ctrlstring])(params[0], wires=wires[:3])]

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "CU(8) gate"



def tracepooling(params, wires):
    qml.CRZ(params[0], wires=[wires[1], wires[0]])
    qml.PauliX(wires=wires[1])
    qml.CRX(params[1], wires=[wires[1], wires[0]])


def measpooling(params, wires):
    m = qml.measure(wires[1])
    qml.cond(m, qml.RZ, qml.RX)(params[int(not m)], wires=wires[0])


class PoolingGate(Operation):
    num_wires = 2
    nparams = 2
    num_params = 1
    grad_method = 'A'
    name = 'Pooling'

    @staticmethod
    def compute_decomposition(*params, wires=None):
            return [tracepooling(params[0], wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "Pooling"