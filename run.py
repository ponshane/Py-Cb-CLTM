from src.CLTM import CLTM
from src.utils import str2bool
import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
 level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--numTopics", type=int, required=True, help="number of topics")
parser.add_argument("--alpha", type=float, required=True, help="dirichlet prior of document-topic distribution")
parser.add_argument("--pathToCorpus", required=True, type=str, help="path of corpus")
parser.add_argument("--vectorFilePath", required=True, type=str, help="path of wordVector")
parser.add_argument("--parallel", type=str2bool, required=True, help="parallel mode?")
parser.add_argument("--num_processes", type=int, required=True, help="number of process")
parser.add_argument("--iters", type=int, required=True, help="number of iterations for sampling process")
args = parser.parse_args()

cltm = CLTM(numTopics=args.numTopics, alpha=args.alpha, pathToCorpus=args.pathToCorpus,
 vectorFilePath=args.vectorFilePath, parallel=args.parallel, num_processes=args.num_processes)

cltm.sample(args.iters)
cltm.dump_pickles()