import jax.lax

import gates
from functools import partial
import itertools
import pennylane as qml
from jax import numpy as np, jit
import numpy
from config import players, separate, classes, pool_type, convolution


#
#
# import numpy as np
#
#
# def jit(args):  # debugging
#     return args


def clayer1(params, w, filters):
    wires = w[::-1][int(numpy.ceil(numpy.log2(filters)))::][::-1].copy()
    wires.append(w[0])
    wires.append(w[1])
    current = 0
    for i in range(len(w) - numpy.ceil(numpy.log2(filters)).astype(int))[::3]:
        gates.U8(params[current:current + gates.U8.nparams], wires=wires[i:(i + 3)])
        current += gates.U8.nparams
        for j in range(1, filters):
            digits = int(numpy.ceil(numpy.log2(filters)))
            ctrlstring = format(j, f'#0{digits}')
            gates.CU8((params[current:current + gates.U8.nparams],), wires=wires[i:(i + 3)] + w[::-1][:digits],
                      ctrlstring=ctrlstring)
            current += gates.U8.nparams
        current -= filters * gates.U8.nparams

    current += convolution * gates.U8.nparams

    for i in range(len(w) - numpy.ceil(numpy.log2(filters)).astype(int))[1::3]:
        gates.U8(params[current:current + gates.U8.nparams], wires=wires[i:(i + 3)])
        current += gates.U8.nparams
        for j in range(1, filters):
            digits = int(numpy.ceil(numpy.log2(filters)))
            ctrlstring = format(j, f'#0{digits}')
            gates.CU8((params[current:current + gates.U8.nparams],), wires=wires[i:(i + 3)] + w[::-1][:digits],
                      ctrlstring=ctrlstring)
            current += gates.U8.nparams
        current -= filters * gates.U8.nparams

    current += convolution * gates.U8.nparams

    for i in range(len(w) - numpy.ceil(numpy.log2(filters)).astype(int))[2::3]:
        gates.U8(params[current:current + gates.U8.nparams], wires=wires[i:(i + 3)])
        current += gates.U8.nparams
        for j in range(1, filters):
            digits = int(numpy.ceil(numpy.log2(filters)))
            ctrlstring = format(j, f'#0{digits}')
            gates.CU8((params[current:current + gates.U8.nparams],), wires=wires[i:(i + 3)] + w[::-1][:digits],
                      ctrlstring=ctrlstring)
            current += gates.U8.nparams
        current -= filters * gates.U8.nparams


cl1params = (3**convolution) * gates.U8.nparams


def clayer2(params, w, filters):
    return clayer1(params, w, filters)


cl2params = cl1params


def clayer3(params, w, filters=1):
    return clayer1(params, w, filters)


cl3params = cl1params

pparams = gates.PoolingGate.nparams


def player(params, wires, filt):
    w = wires[::-1][int(numpy.log2(filt)):][::-1].copy()
    current = 0
    w.append(w[0])
    for i in range(1, len(wires) - int(numpy.log2(filt)), 2):
        gates.PoolingGate(params[current:current + pparams], wires=w[i - 1:i + 1])
        if separate:
            current += pparams
        digits = int(numpy.ceil(numpy.log2(filt)))
        for j in range(1, filt):
            ctrlstring = format(j, f'#0{digits}')
            qml.ctrl(gates.PoolingGate, wires[::-1][:digits], control_values=[bool(int(i)) for i in ctrlstring]) \
                (params[current:current + pparams], wires=w[i - 1:i + 1])
            if separate:
                current += pparams

    # for i in range(1, len(wires) - int(numpy.log2(filt)), 2):
    #     gates.PoolingGate(params[current:current + pparams], wires=w[i + 1:i - 1:-1])
    #     if separate:
    #         current += pparams
    #     digits = int(numpy.ceil(numpy.log2(filt)))
    #     for j in range(1, filt):
    #         ctrlstring = format(j, f'#0{digits}')
    #         qml.ctrl(gates.PoolingGate, wires[::-1][:digits], control_values=[bool(int(i)) for i in ctrlstring]) \
    #             (params[current:current + pparams], wires=w[i + 1:i - 1:-1])
    #         if separate:
    #             current += pparams


def fclayer(params, wires, read):
    current = 0

    # for i in range(len(wires)-2):
    #     for j in range(i+1, len(wires)-1):
    #         for l in range(j+1, len(wires)):
    #             gates.U8(params[current:current+82], wires=[wires[i], wires[j], wires[l]])
    #             current += 82

    # clayer1(params[current:current+cl1params], wires, filters=1)

    # for i in range(read, len(wires)):
    #     for j in range(read):
    #         gates.PoolingGate(params[current:current + pparams], wires=[wires[j], wires[i]])
    #         current += pparams
    #
    # for i in range(1, len(wires)):
    #     for j in range(i + 1):
    #         for l in range(j + i, len(wires), i + 1):
    #             gates.U4Gate(params[current:current + gates.U4Gate.nparams], wires=[wires[l - i], wires[l]])
    #             current += gates.U4Gate.nparams
    #
    # for i in range(1, len(wires)):
    #     for j in range(i + 1):
    #         for l in range(j + i, len(wires), i + 1):
    #             gates.U4Gate(params[current:current + gates.U4Gate.nparams], wires=[wires[l - i], wires[l]])
    #             current += gates.U4Gate.nparams

    # for i in itertools.combinations(wires, 4):
    #     gates.U16gate(params[current:current+66], wires=list(i))
    #     current += 66

    # nextpar = int(15*len(wires)*(len(wires)-1)/2)
    # qml.broadcast(unitary=gates.U4Gate, wires=wires, pattern="all_to_all", parameters=np.reshape(
    #     params[current:current+nextpar], (-1, 1, 15)))
    # current += nextpar
    #
    # qml.broadcast(unitary=gates.U4Gate, wires=wires, pattern="all_to_all", parameters=np.reshape(
    #     params[current:current+nextpar], (-1, 1, 15)))
    # current += nextpar

    # for i in range(read, len(wires)):
    #     for j in range(read):
    #         gates.PoolingGate(params[current:current + pparams], wires=[wires[j], wires[i]])
    #         current += pparams

    # nextpar = int(15 * read * (read - 1) / 2)
    # qml.broadcast(unitary=gates.U4Gate, wires=wires[read-1::-1], pattern="all_to_all", parameters=np.reshape(
    #     params[current:current+nextpar], (-1, 1, 15)))
    # current += nextpar
    #
    # qml.broadcast(unitary=gates.U4Gate, wires=wires[read - 1::-1], pattern="all_to_all", parameters=np.reshape(
    #     params[current:current + nextpar], (-1, 1, 15)))

    # for i in range(1, read):
    #     for j in range(i + 1):
    #         for l in range(j + i, read, i + 1):
    #             gates.U4Gate(params[current:current + gates.U4Gate.nparams], wires=[wires[l - i], wires[l]])
    #             current += gates.U4Gate.nparams

    # for i in range(read-2):
    #     for j in range(i+1, read-1):
    #         for l in range(j+1, read):
    #             gates.U8(params[current:current+82], wires=[wires[i], wires[j], wires[l]])
    #             current += 82

    # for i in range(1, read):
    #     for j in range(i + 1):
    #         for l in range(j + i, read, i + 1):
    #             gates.U4Gate(params[current:current + 15], wires=[wires[l - i], wires[l]])
    #             current += 15
    #
    # for i in range(1, read):
    #     for j in range(i + 1):
    #         for l in range(j + i, read, i + 1):
    #             gates.U4Gate(params[current:current + 15], wires=[wires[l - i], wires[l]])
    #             current += 15

    # for i in range(read-1, 0, -1):
    #     for j in range(i+1):
    #         for l in range(j+i, read, i+1):
    #             gates.U4Gate(params[current:current+15], wires=[wires[l-i], wires[l]])
    #             current += 15


def fcparams(nwires, read):
    fcp = 0
    fcp += 0 * gates.U4Gate.nparams * read * (read - 1) / 2  # U4 gates on read qubits
    fcp += 0 * gates.U4Gate.nparams * nwires * (nwires - 1) / 2  # U4 gates on nwires qubits
    fcp += 0 * pparams * read * (nwires - read)  # pooling gates from nwires - read qubits to read qubits
    # fcp += gates.U8.nparams * read * (read-1) * (read-2) / 6  # U8 gates on read qubits
    # fcp = 2*read*(nwires-read) + 4**read - 1
    # fcp = 4*read*(nwires-read) + 66*read*(read-1)*(read-2)*(read-3)/24 + 66*nwires*(nwires-1)*(nwires-2)*(nwires-3)/24
    # fcp = 1*pparams*read*(nwires-read) + 4**read - 1
    return fcp


def circuit(vec, params, filters):
    read = int(numpy.ceil(numpy.log2(classes)))
    inputs = int(numpy.ceil(numpy.log2(len(vec))))  # number of input qubits
    activewires = list(range(inputs))
    currentparam = 0

    qml.AmplitudeEmbedding(features=vec, wires=range(inputs), normalize=True, pad_with=0)

    while len(activewires) > 2 * read:
        clayer1(params[currentparam:currentparam + filters * cl1params], activewires, filters)
        currentparam = currentparam + cl1params * filters
        player(params[currentparam:currentparam + players * pparams * (filters * (len(activewires) // 2)) ** separate],
               activewires, filters)
        currentparam = currentparam + players * pparams * (filters * (len(activewires) // 2)) ** separate
        activewires = activewires[::2]
        break
        if len(activewires) <= 2 * read:
            break
        clayer2(params[currentparam:currentparam + filters * cl2params], activewires, filters)
        currentparam = currentparam + cl2params * filters

        player(params[currentparam:currentparam + players * pparams * (filters * (len(activewires) // 2)) ** separate],
               activewires, filters)
        currentparam = currentparam + players * pparams * (filters * (len(activewires) // 2)) ** separate
        activewires = activewires[::2]
        if len(activewires) <= 2 * read:
            break
        clayer3(params[currentparam:currentparam + cl3params], activewires)
        currentparam = currentparam + cl3params

        player(params[currentparam:currentparam + players * pparams * (filters * (len(activewires) // 2)) ** separate],
               activewires, filters)
        currentparam = currentparam + players * pparams * (filters * (len(activewires) // 2)) ** separate
        activewires = activewires[::2]

    fclayer(params[currentparam:], activewires, read)

    return qml.probs(activewires[:read])


def numparams(active, classes, filt):
    read = int(np.ceil(np.log2(classes)))
    nparams = 0

    while active >= 2 * read:
        nparams += cl1params * filt + players * filt * pparams * (1 * (active // 2) ** separate)
        active = np.ceil(active / 2)
        break
        if active < 2 * read:
            break
        nparams += cl2params * filt

        nparams += players * filt * pparams * (1 * (active // 2) ** separate)
        active = np.ceil(active / 2)
        if active < 2 * read:
            break
        nparams += cl3params * filt

        nparams += players * filt * pparams * (1 * (active // 2) ** separate)
        active = np.ceil(active / 2)

    nparams = int(nparams + fcparams(active, read))

    return nparams
