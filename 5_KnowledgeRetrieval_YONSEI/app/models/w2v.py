import json
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim import matutils


class Retrieval_w2v(object):
    def __init__(self):
        config = json.load(open("./data/w2v/config_w2v.json"))

        model_path = config['model']
        self.model = Word2Vec.load(model_path)

        probsfile_path = config['probsfile']

        self.entityv = KeyedVectors(self.model.vector_size * 2)
        entityv_entities = []
        entityv_weights = []
        self.wordv = KeyedVectors(self.model.vector_size * 2)
        wordv_entities = []
        wordv_weights = []

        size = -1
        for entity, idx in self.model.wv.key_to_index.items():
            size += 1
            if size > 10000:
                continue
            if entity.startswith('<'): 
                entityv_entities.append(entity)
                entityv_weights.append(np.concatenate((self.model.syn1neg[idx], self.model.wv.vectors[idx])))

            else:
                wordv_entities.append(entity)
                wordv_weights.append(np.concatenate((self.model.wv.vectors[idx], self.model.syn1neg[idx])))


        self.entityv.add_vectors(entityv_entities, entityv_weights)
        self.entityv.fill_norms()

        self.wordv.add_vectors(wordv_entities, wordv_weights)
        
        self.word_probs = {}
        with open(probsfile_path) as f:
            for line in f:
                word, prob = line.rstrip('\n').split('\t')
                self.word_probs[word] = float(prob)


    def wmean(self, question_txt):

        positive = []
        question_txt = question_txt.split()
        for token in question_txt:
            token = token.lower()

            if token in self.wordv.key_to_index:
                if token in self.word_probs:
                    weight = 0.0003 / (0.0003 + self.word_probs[token])
                    positive.append(self.wordv.get_vector(token, norm=False) * weight)
                else:
                    pass

        if not positive:
            positive.append(np.zeros(self.entityv.vector_size))
            
        mean = matutils.unitvec(np.array(positive).mean(axis=0))

        return mean


    def retrieve(self, question_txt, ent_list_lda):

        qmean = self.wmean(question_txt)

        ent_list = []
        ent_set = set(ent_list_lda)
        for entity in ent_set:
            ent_url = "<http://dbpedia.org/resource/"+entity+">"
            if ent_url in self.entityv.key_to_index:
                emb_score = np.dot(qmean, self.entityv.get_vector(ent_url, norm=True))
                if emb_score > 0:
                    ent_list.append(entity)

        return ent_list
