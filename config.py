import optax
import argparse
from jax import devices

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trainsets", type=int, help="number of training sets", default=240000)
parser.add_argument("-i", "--iterations", type=int, help="numbers of iterations", default=[4000], nargs="+")
parser.add_argument("-r", "--rates", type=float, help="learning rates", default=[0.006], nargs="+")
parser.add_argument("-b", "--batchsizes", type=int, help="batch sizes", default=[64], nargs="+")
parser.add_argument("-f", "--filters", type=int, help="numbers of filters", default=[1], nargs="+")
parser.add_argument("-n", "--trainingruns", type=int, help="number of independent training runs", default=10)
parser.add_argument("-s", "--separate", type=bool, help="separate parameters for pooling gates?", default=False)
parser.add_argument("-p", "--trainperf", type=bool, help="evaluate train performance?", default=False)
parser.add_argument("-tb", "--testbatch", type=int, help="test batch size", default=40)
parser.add_argument("-pt", "--pooltype", type=str, help="pooling type meas or trace", default='trace')
parser.add_argument("-c", "--classes", type=int, help="number of classes", default=2)
parser.add_argument("-d", "--dataset", type=str, help="dataset can be 'balanced', 'byclass', 'bymerge', 'digits', "
                                                      "'letters', 'mnist', 'kuzushiji', 'fashion'", default='mnist')
parser.add_argument("-ct", "--convolution_type", type=bool, help='0 if 82 pars, 1 if 246', default=0)


args = parser.parse_args()

samples = args.dataset  # dataset
train_sets = args.trainsets  # number of training train_sets
iterations = args.iterations  # number of training epochs
rates = args.rates  # learning rate
batch_sizes = args.batchsizes  # batch size in each epoch
filters = args.filters  # number of convolutional filters
ntrials = args.trainingruns  # number of independent training runs
separate = args.separate  # 1 if pooling gates have independent parameters
evaluate_train_performance = args.trainperf  # evaluate performance on training samples?
test_batch = args.testbatch  # batch size while evaluating performance
pool_type = args.pooltype  # pooling type
classes = args.classes  # number of classes
convolution = args.convolution_type

optim = "Adam"  # name of optimizer, for labels
optimizer = optax.adam  # set optimizer for optax
randinitialization = 1  # initialize parameters randomly?
players = 1  # number of pooling operations each qubit undergoes per pool layer(edit gates.py accordingly)
parent_dir = r'./app'  # directory to output to
