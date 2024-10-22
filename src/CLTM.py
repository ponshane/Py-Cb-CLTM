import logging
import time
import pickle
import numpy as np
from scipy.optimize import minimize
from .utils import normalize_rows
from numba import jit

class CLTM(object):
    def __init__(self, numTopics, alpha, pathToCorpus, vectorFilePath):
        self.word2IdVocabulary = dict()
        self.id2WordVocabulary = dict()
        self.numWordsInCorpus = 0

        self.numTopics = numTopics
        self.alpha = alpha
        self.read_corpus(pathToCorpus)
        logging.info("Loaded Corpus.")

        ### initialize parameters of the model
        self.vocabularySize = len(self.word2IdVocabulary)
        self.numDocuments = len(self.corpus)
        self.docTopicCount = np.zeros((self.numDocuments, self.numTopics)) 
        self.sumDocTopicCount = np.zeros(self.numDocuments)
        self.topicWordCountLF = np.zeros((self.numTopics, self.vocabularySize)) 
        self.sumTopicWordCountLF = np.zeros(self.numTopics) 

        self.alphaSum = self.numTopics * self.alpha
        
        self.readWordVectorsFile(vectorFilePath)
        logging.info("Loaded Wordvectors.")
        self.initialize()
        logging.info("Initialized params of the model.")

    def read_corpus(self, pathToCorpus):
        self.corpus = list()
        with open(pathToCorpus, "r") as fh:
            for doc_in_file in fh:
                document = list()
                for word in doc_in_file.strip("\n").split(" "):
                    if word not in self.word2IdVocabulary.keys():
                        dict_length = len(self.word2IdVocabulary)
                        self.word2IdVocabulary[word] = dict_length
                        self.id2WordVocabulary[dict_length] = word
                        # word2id
                        document.append(self.word2IdVocabulary[word])
                    else:
                        document.append(self.word2IdVocabulary[word])
                # push document into corpus
                self.numWordsInCorpus = self.numWordsInCorpus + len(document)
                self.corpus.append(document)
    
    def readWordVectorsFile(self, vectorFilePath):
        with open(vectorFilePath, "r") as fh:

            """
            Take first line to infer vectorSize and then initialize wordVectors
            """
            line = fh.readline()
            elements = line.strip("\n").split(" ")
            self.vectorSize = len(elements) - 1
            self.wordVectors = np.zeros((self.vocabularySize, self.vectorSize))
            word = elements[0]
            if word in self.word2IdVocabulary.keys():
                self.wordVectors[self.word2IdVocabulary[word],:] = elements[1:]
            
            """
            Other word vectors
            """
            while line:
                line = fh.readline()
                elements = line.strip("\n").split(" ")
                word = elements[0]
                if word in self.word2IdVocabulary.keys():
                    self.wordVectors[self.word2IdVocabulary[word],:] = elements[1:]

        self.wordVectors = normalize_rows(self.wordVectors)

    def initialize(self):
        self.topicAssignments = list()
        multiPros = [1/self.numTopics] * self.numTopics
        for docId in range(self.numDocuments):
            topics = list()
            docSize = len(self.corpus[docId])
            for j in range(docSize):
                wordId = self.corpus[docId][j]
                # randomly assign topic index for each word
                topic = np.random.choice(a=self.numTopics,size=1,p=multiPros)[0]
                self.topicWordCountLF[topic, wordId] += 1
                self.sumTopicWordCountLF[topic] += 1    
                self.docTopicCount[docId, topic] += 1
                self.sumDocTopicCount[docId] += 1
                topics.append(topic)
            
            self.topicAssignments.append(topics)
        self.topicVectors = np.random.rand(self.numTopics, self.vectorSize)
    
    def sample(self, Iteration):
        expDotProductValues = np.zeros((self.numTopics, self.vocabularySize))
        sumExpValues = np.zeros(self.numTopics)
        logging.info("Start to optimize:")
        for i in range(Iteration):
            after_cost = 0
            start = time.time()
            for t_index in range(self.numTopics):
                oldtopicVec = self.topicVectors[t_index,:]
                solution = minimize(fun=self.Loss,
                    x0=self.topicVectors[t_index,:], args=(t_index), method="L-BFGS-B",
                    jac=self.gradient_func,
                    options={'gtol': 1e-3, 'disp': False})
                after_cost += solution["fun"]
                newtopicVec = solution["x"]
                self.topicVectors[t_index,:] = newtopicVec
                expDotProductValues[t_index, :] = np.exp(np.dot(newtopicVec,self.wordVectors.T))
                sumExpValues[t_index] = np.sum(expDotProductValues[t_index, :])
            logging.info("After {} Iters, Avg. Cost = {}, Elaspsed Seconds: {}".format(i,
             after_cost/self.numTopics, time.time() - start))
            for docId in range(self.numDocuments):
                docSize = len(self.corpus[docId])
                for j in range(docSize):
                    wordId = self.corpus[docId][j]
                    old_topic = self.topicAssignments[docId][j]
                    self.topicWordCountLF[old_topic, wordId] -= 1
                    self.sumTopicWordCountLF[old_topic] -= 1    
                    self.docTopicCount[docId, old_topic] -= 1
                    
                    multiPros = [0]*self.numTopics
                    for t_index in range(self.numTopics):
                        multiPros[t_index] = (self.docTopicCount[docId, t_index] + self.alpha) *\
                            (expDotProductValues[t_index, wordId]/sumExpValues[t_index])
                    tot = sum(multiPros)
                    multiPros = [elem/tot for elem in multiPros]
                    new_topic = np.random.choice(a=self.numTopics,size=1,p=multiPros)[0]
                    self.topicAssignments[docId][j] = new_topic
                    self.topicWordCountLF[new_topic, wordId] += 1
                    self.sumTopicWordCountLF[new_topic] += 1    
                    self.docTopicCount[docId, new_topic] += 1
    
    @staticmethod
    @jit(nopython=True)
    def _compiled_loss(topicWordCountLF, wordVectors, vec, idx):
        return -np.dot(topicWordCountLF[idx,:],\
            (np.dot(vec, wordVectors.T) -\
                 np.log(np.sum(np.exp(np.dot(vec, wordVectors.T))))))

    def Loss(self, vec, idx):
        return self._compiled_loss(self.topicWordCountLF, self.wordVectors, vec, idx)
    
    @staticmethod
    @jit(nopython=True)
    def _compiled_gradient_func(wordVectors, vec, topicWordCountLF, idx):
        expection = np.exp(np.dot(vec, wordVectors.T)).T / np.sum(np.exp(np.dot(vec, wordVectors.T))) # with shape = V x 1
        weighted_dims = np.sum(wordVectors * np.expand_dims(expection, axis=1), # expection[:,np.newaxis] = np.expand_dims(expection, axis=1)
         axis=0) # with shape 1 x d
        grad = np.zeros(wordVectors.shape)
        for word_ind in range(wordVectors.shape[0]):
            grad[word_ind,:] = (wordVectors[word_ind,:] - weighted_dims) * topicWordCountLF[idx,word_ind]
        return np.sum(grad, axis=0) * (-1)

    def gradient_func(self, vec, idx):
        return self._compiled_gradient_func(self.wordVectors, vec, self.topicWordCountLF, idx)
    
    def export_topics(self, top_n = 500):
        """ export topic words

        Args:
            top_n (int, optional): number of words per topic. Defaults to 500.

        Returns:
            list: list of topic words, each indicates each topics
        """
        list_of_topics = []
        for t_index in range(self.numTopics):
            expWordProduct = np.exp(np.dot(self.topicVectors[t_index,:],self.wordVectors.T))
            topicWordIndexes = (expWordProduct / np.sum(expWordProduct)).argsort()[::-1][:top_n]
            topicWords = [self.id2WordVocabulary[i] for i in topicWordIndexes]
            list_of_topics.append(topicWords)
        return list_of_topics

    def export_docTopics(self):
        self.docTopicCount = self.docTopicCount + self.alpha
        normDocTopics = self.docTopicCount / \
            np.expand_dims(np.sum(self.docTopicCount, axis=1), axis=1)
        return normDocTopics

    def dump_pickles(self, file_path="./tmp.pickle", top_n=500):
        list_of_topics = self.export_topics(top_n = top_n)
        normDocTopics = self.export_docTopics()
        with open(file_path, "wb") as fh:
            pickle.dump((self.topicVectors, list_of_topics, normDocTopics), fh)