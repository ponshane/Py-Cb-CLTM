from src.CLTM import CLTM
import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
 level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--numTopics", type=int, required=True, help="number of topics")
parser.add_argument("--alpha", type=float, required=True, help="dirichlet prior of document-topic distribution")
parser.add_argument("--pathToCorpus", required=True, type=str, help="path of corpus")
parser.add_argument("--vectorFilePath", required=True, type=str, help="path of wordVector")
parser.add_argument("--iters", type=int, required=True, help="number of iterations for sampling process")
parser.add_argument("--prefix", type=str, default="TEST", help="prefix name of output pickle")
args = parser.parse_args()

cltm = CLTM(numTopics=args.numTopics, alpha=args.alpha, pathToCorpus=args.pathToCorpus,
 vectorFilePath=args.vectorFilePath)

cltm.sample(args.iters)
cltm.dump_pickles(file_path=f"./{args.prefix}-a{args.alpha}-iter{args.iters}.pkl")