import json
import pickle
import gensim


class Retrieval_lda(object):
    def __init__(self):
        config = json.load(open("./data/lda/config_lda.json"))

        dictionary_path = config['dictionary']
        tokenized_doc_path = config['tokenized_doc']
        entity_tag_dict_path = config['entity_tag_dict']
        
        self.queries_path = config['queries_path']
        self.model_path = config['model']

        with open(dictionary_path, 'rb') as f:
            self.dictionary = pickle.load(f)
        with open(tokenized_doc_path, 'rb') as f:
            self.tokenized_doc = pickle.load(f)
        with open(entity_tag_dict_path, 'rb') as f:
            self.entity_tag_dict = pickle.load(f)

        
    def retrieve(self, question_txt, ent_list):

        lda_model = gensim.models.ldamodel.LdaModel.load(self.model_path)
        query = question_txt.replace("?", "").split()
        query_terms = []
        for q in query:
            if q in self.dictionary.token2id:
                query_terms.append(q)
            else:
                pass
        
        if len(query_terms) == 0:
            return ent_list
        
        query_results = []
        for entity in ent_list:
            score = 0
            
            if entity in self.entity_tag_dict.keys():
                doc = self.tokenized_doc[list(self.entity_tag_dict.keys()).index(entity)]
                doc = self.dictionary.doc2bow(doc)

                document_topic_dist = lda_model.get_document_topics(doc)

                if len(document_topic_dist) == 0:
                    pass
                else:
                    for doc_topic in document_topic_dist:
                        doc_topic_id = doc_topic[0]
                        prob_topic_doc = doc_topic[1]
                        
                        for q in query_terms:
                            word_topic_dist = lda_model.get_term_topics(self.dictionary.token2id[q])
                            
                            if len(word_topic_dist) == 0:
                                pass
                            else:
                                for word_topic in word_topic_dist:
                                    word_topic_id = word_topic[0]
                                    prob_word_topic = word_topic[1]
                                    
                                    if word_topic_id == doc_topic_id:
                                        score += prob_topic_doc * prob_word_topic
                                    else:
                                        pass
            else:
                pass
            
            if score != 0:
                query_results.append(entity)


        if len(query_results) != 0:
            return query_results
        else:
            return ent_list

