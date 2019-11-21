from src.CLTM import CLTM
from scipy.optimize import minimize
import multiprocessing.pool
import logging
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
 level=logging.INFO)

cltm = CLTM(numTopics=10, alpha=0.1,
 beta=0.01, pathToCorpus="/home/ponshane/work_dir/CLTM-Experiments/Data/UM-Corpus/50K-sampled-docs/selected50KDos.txt",
 vectorFilePath="/home/ponshane/work_dir/CLTM-Experiments/Data/UM-Corpus/word-vectors/80dim_50K_selected_vectors.txt")

cltm.sample(5)
print()

# single process
# start = time.time()
# initial_cost = 0
# after_cost = 0
# for i in range(cltm.numTopics):
#     initial_cost += cltm.Loss(cltm.topicVectors[i,:], i)

#     solution = minimize(fun=cltm.Loss,
#     x0=cltm.topicVectors[i,:], args=(i), method="BFGS",
#     jac=cltm.gradient_func,
#     options={'gtol': 1e-3, 'disp': False})
#     after_cost += solution["fun"]
#     print("finish: ", i)
# end = time.time()

# start = time.time()
# initial_cost = 0
# after_cost = 0
# args = [i for i in range(cltm.numTopics)]
# p = multiprocessing.pool.ThreadPool()
# results = p.map(cltm.minimize_parallel,args)
# end = time.time()

# print(end-start)