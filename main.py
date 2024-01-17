import os, argparse, optax
import jax
import numpy
import numpy as np

from matplotlib import pyplot as plt

from loader import *
from training import *
from config import *
from scipy import interpolate

# import numpy as np
#
#
# def jit(args):  # debugging
#     return args


numpy.set_printoptions(suppress=True, precision=1, linewidth=200)


i = 1
while True:
    try:
        localdir = f"Size_{numpy.amin(batch_sizes)}to{numpy.amax(batch_sizes)}_" \
                   f"Filt_{numpy.amin(filters)}to{numpy.amax(filters)}_" \
                   f"Rate_{round(numpy.amin(rates), 3)}to{round(numpy.amax(rates), 3)}_" \
                   f"Iters_{numpy.amin(iterations)}to{numpy.amax(iterations)}_Run_{i}"
        directory = os.path.join(parent_dir, localdir)
        os.makedirs(directory)
        break
    except FileExistsError:
        i += 1


seed = int(time())
key = jax.random.PRNGKey(seed)
print(seed)


for batch_size, filt in itertools.product(batch_sizes, filters):
    trainvecs, trainlbls, testimgs, testlbls = load(samples, train_sets, filt)
    inputs = int(np.log2(len(trainvecs[0])))  # number of input qubits
    dev = qml.device("default.qubit.jax", wires=inputs)
    qnode = qml.QNode(circuit, dev, interface='jax-jit')
    nparams = numparams(inputs, classes, filt)  # number of params
    print(f"{nparams} parameters")
    for iteration, rate in itertools.product(iterations, rates):
        trial = 0
        # avgaccuracy = 0
        # avgtrainacc = 0
        if evaluate_train_performance == 1:
            trainaccs = numpy.zeros(ntrials)
        accuracies = numpy.zeros(ntrials)
        while trial < ntrials:
            i = 1
            while True:
                try:
                    localdir = f"Size_{batch_size}_Filt_{filt}_Rate_{rate}_Iters_{iteration}_Run_{i}"
                    trialdirectory = os.path.join(directory, localdir)
                    os.makedirs(trialdirectory)
                    break
                except FileExistsError:
                    i += 1

            if randinitialization:
                params = 4 * np.pi * jax.random.uniform(key, [nparams]).astype(float)
            else:
                params = np.zeros(nparams).astype(float)
            opt_state = optimizer(rate).init(params)
            costs = []
            fig, ax = qml.draw_mpl(qnode)(trainvecs[0], params, filt)
            plt.savefig(f'{trialdirectory}/Circuit.pdf')
            plt.clf()
            plt.cla()
            plt.close(fig)
            # f = open(f"{trialdirectory}/circuit.txt", "a", encoding="utf-8")
            # f.write(qml.draw(qnode, decimals=None)(trainvecs[0], params, filt))
            print(f'Circuit drawing saved at {int(time()) - seed} seconds')
            print(trialdirectory)

            for i in range(iteration):
                key, subkey = jax.random.split(key)
                batch = jax.random.choice(subkey, trainvecs, [batch_size], False)
                batch_correct = jax.random.choice(subkey, trainlbls, [batch_size], False)
                grad, stepcost = gradient_and_cost(batch, batch_correct, params, batch_size, filt)
                # grad, stepcost = gradient_and_cost(batch, batch_correct, params, batch_size, filt)
                updates, opt_state = optimizer(rate).update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                costs.append(stepcost)
                if not i % 100:
                    print(f'{i} at {int(time()) - seed} seconds')

            plt.plot(costs, alpha=0.5)
            plt.plot(savitzky_golay(np.array(costs), 101, 5), color='red')
            plt.title(f'{nparams}-parameter {optim} optimizer, {rate} learning rate,\n'
                      f'{iteration} iterations, batch size {batch_size}')
            plt.xlabel('Iterations')
            plt.ylabel('Cross-entropy loss')
            plt.savefig(f'{trialdirectory}/Costs.png')
            print(f'Figure saved at {int(time()) - seed} seconds')
            plt.close()

            f = open(f"{trialdirectory}/Params.txt", "a")
            f.write(f'{nparams}-parameter {optim} Optimizer\n'
                    f'\nTrain size - {train_sets}\tLearning rate - {rate}\tIterations - {iteration}\tBatch size - {batch_size}\n'
                    f'Seed - {seed}\n{nparams} Parameters - \n {params}\n')
            f.close()

            accuracies[trial], correctprobs, classaccuracy, confusion = test_accuracy(testimgs, testlbls, params, classes, filt)
            # avgaccuracy = (trial * avgaccuracy + accuracies) / (trial + 1)
            f = open(f'{trialdirectory}/Results.txt', 'a')
            f.write(f'{nparams}-parameter {optim} Optimizer\n'
                    f'\nLearning rate - {rate}\tIterations - {iteration}\tBatch size - {batch_size}\tFilters - {filt}\n'
                    f'Performance on test cases - \n'
                    f'Max correct probability - {np.max(correctprobs)}\tMin correct probability - {np.min(correctprobs)}\n'
                    f'Avg correct probability - {np.average(correctprobs)}\t Deviation - {np.std(correctprobs)}\n'
                    f'Accuracy - {accuracies[trial]}\n Accuracy by class -{classaccuracy}\n Confusion matrix - \n{confusion}\n'
                    f'Accuracy over {trial} trials - {np.average(accuracies[:trial+1])}\t'
                    f'Standard deviation - {np.std(accuracies[:trial+1])}\n Accuracies - {accuracies}')
            f.close()
            if evaluate_train_performance == 1:
                trainaccs[trial], traincorr, trainclassacc, trainconfusion = \
                    test_accuracy(trainvecs, trainlbls, params, classes, filt)
                # avgtrainacc = (trial * avgtrainacc + trainaccs[tr]) / (trial + 1)
                f = open(f'{trialdirectory}/Results.txt', 'a')
                f.write(f'Performance on train cases - \n'
                        f'Max correct probability - {np.max(traincorr)}\tMin correct probability - {np.min(traincorr)}\n'
                        f'Avg correct probability - {np.average(traincorr)}\t Deviation - {np.std(traincorr)}\n'
                        f'Accuracy - {trainaccs[trial]}\n Accuracy by class -{trainclassacc}\n '
                        f'Confusion matrix - \n{trainconfusion}\n'
                        f'Accuracy over {trial} trials - {np.average(trainaccs[:trial + 1])}\t'
                        f'Standard deviation - {np.std(trainaccs[:trial + 1])}\n Accuracies - {trainaccs}')
                f.close()
            print(f'{int(time()) - seed} seconds')
            trial += 1

