import numpy as np
import tone_normalization 
from utils import *
import scipy

class SentenceGraph:
    def __init__(self, sentences_list, w2v, use_lm, lm_model, lm_tokenizer, ita=0.9, threshold=0.65):
        self.sentences_list = sentences_list

        self.length = len(sentences_list)

        self.w2v = w2v

        self.use_lm = use_lm 

        self.lm_model = lm_model

        self.tokenizer = lm_tokenizer

        # threshold for step1
        self.threshold = threshold

        # threshold for step4
        self.ita = ita

    def check_noun(self, str1, str2):
        flag = check_noun(str1, str2)
        # check_pronoun(str1, str2) or 
        return flag

    def check_discourse_markers(self, str):
        flag = check_discourse_markers(str)
        return flag

    def compare_name_entity(self, str1, str2):
        flag = compare_name_entity(str1, str2)
        return flag

    # compute the cos similarity between a and b. a, b are numpy arrays
    def cos_sim(self, a, b):
        return 1 - scipy.spatial.distance.cosine(a,b)


    def make_graph_undirected(self, source, target, weight):
        source.extend(target)
        target.extend(source)
        weight.extend(weight)
        triplet_list=[ (source[i],target[i],weight[i]) for i in range(len(source))]
        sorted_by_src = sorted(triplet_list, key=lambda x: (x[0],x[1]))

        sorted_source = []
        sorted_target = []
        sorted_weight = []
        for triplet in sorted_by_src:
            sorted_source.append(triplet[0])
            sorted_target.append(triplet[1])
            sorted_weight.append(triplet[2])
        return sorted_source, sorted_target, sorted_weight

    # Step4: calculate sentence embeddings
    def get_sentence_embeddings(self,list_segement):
        v = np.zeros([self.length,self.size])
        if not self.use_lm:
            for i in range(self.length):
                emb_sen = self.get_wv_embedding(list_segement[i])
                v[i,] = emb_sen
            # v = self.get_wv_embedding(string)
        else:
            v = self.get_lm_embedding(list_segement)
        return v

    # get sentence embeddings with w2v
    def get_wv_embedding(self, sent):
        word_embeddings = self.w2v
        eps = 1e-10
        if len(sent) != 0:
            vectors = [word_embeddings.get(tone_normalization.replace_all(w[0].lower()), np.zeros((300,))) for w in sent]
            v = np.mean(vectors, axis=0)
        else:
            v = np.zeros((300,))
        v = v + eps
        return v    

    # get language model embedding
    def get_lm_embedding(self, list_segement):
        v = self.lm_model.encode([x.lower() for x in list_segement])
        return np.array(v)

    # step 4: compare sentence similarity
    def check_if_similar_sentences(self,sentence_emb1,sentence_emb2):
        flag = False
        similarity = self.cos_sim(sentence_emb1,sentence_emb2)
        if similarity > self.ita:
            flag = True
        return flag


    def build_sentence_graph(self,):
        # spectral clustering  
        X = np.zeros([self.length, self.length])
        
        # get the vector size
        if self.use_lm == True:
            self.size = 768
        else: 
            self.size = 300
        # self.size = len(self.get_sentence_embeddings(self.sentences_list[0]))

        # get sentence vector holder
        emb_sentence_vectors = np.zeros([self.length,self.size])
    
        list_segement = []
        for i in range(self.length):
            tmp = segment(self.sentences_list[i])
            list_segement.append(tmp)
        
        if self.use_lm == True:
            emb_sentence_vectors = self.get_sentence_embeddings(self.sentences_list)
        else: 
            emb_sentence_vectors = self.get_sentence_embeddings(list_segement)

        # print(emb_sentence_vectors.shape)
        # for i in range(self.length):
        #      emb_sen = self.get_sentence_embeddings(list_segement[i])
        #      emb_sentence_vectors[i,] = emb_sen

        # iterate all sentence nodes to check if they should be connected
        for i in range(self.length):
            for j in range(i+1,self.length):
                flag = False
                if (j-i) == 1:
                    flag = False
                    # flag = self.check_noun(list_segement[i], list_segement[j])
                    if not flag:
                        # check for disourse markers
                        flag = self.check_discourse_markers(list_segement[j][0])
                else:
                    # check for name entities
                    flag=self.compare_name_entity(list_segement[i], list_segement[j])

               # => step4 check for similar sentences
                tmp = 0
                if not flag:
                    # continue
                    i_sen_emb = emb_sentence_vectors[i,]
                    j_sen_emb = emb_sentence_vectors[j,]
                    flag = self.check_if_similar_sentences(i_sen_emb,j_sen_emb)
                    # tmp = self.cos_sim(i_sen_emb,j_sen_emb)     

                if flag:                               
                    X[i,j] = 1
                    X[j,i] = 1
        return X