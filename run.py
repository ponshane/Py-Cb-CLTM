from src.CLTM import CLTM
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
 level=logging.INFO)

cltm = CLTM(numTopics=10, alpha=0.1,
 beta=0.01, pathToCorpus="/home/ponshane/work_dir/CLTM-Experiments/Data/UM-Corpus/50K-sampled-docs/selected50KDos.txt",
 vectorFilePath="/home/ponshane/work_dir/CLTM-Experiments/Data/UM-Corpus/word-vectors/80dim_50K_selected_vectors.txt",
 parallel=True, num_processes=2)

cltm.sample(2)
cltm.dump_pickles()