import json
import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from scipy.spatial.distance import pdist, squareform
from bpemb import BPEmb
import re
from collections import defaultdict



class Service:
    task = [
        {
            'name': 'Knowledge_Merging', 
            'description':'Merge knowledge graph and supporting graph from documents'
        }
    ]

    def __init__(self):
        self.graph_merge = Merging_supporting_graphs()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bpemb_en = BPEmb(lang="en", dim=100, vs=200000)

    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            # ret = content
            ret = self.graph_merge.do_search(content)

            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400

class Merging_supporting_graphs(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bpemb_en = BPEmb(lang="en", dim=100, vs=200000)

    def do_search(self, content):
        # question = content.get('question', None)
        kg_triples = content.get('kg_triples', None)
        doc_triples = content.get('doc_triples', None)

        # with open("../../5_KnowledgeRetrieval_YONSEI/app/data/supporting_facts_from_kg.json") as f:
        #     content = json.load(f)
        # kg_triples = content.get('kg_triples', None)

        # if question is None:
        #     return {
        #         'error': "invalid question"
        #     }
        # elif not os.path.exists(graph_path):
        #     return {
        #         'error': "knowledge graph file doesn't exist"
        #     }
        # else:
        #     question_txt = question.get('text', '')
        #     question_ln = question.get('language', 'kr')
        #     question_domain = question.get('domain', 'common-sense')

            # if question_txt == '':
            #     return {
            #         'error': "invalid question text"
            #     }
            # else:
            #     return self.search(kg_triples, doc_triples)

        if kg_triples is None:
            return {
                'error': "invalid kg_triples"
            }
        
        if doc_triples is None:
            return {
                'error': "invalid doc_triples"
            }        

        return self.search(kg_triples, doc_triples)




    def word_tokenizer(self, word):
        tokenized_word = []

        if re.findall("[0-9]+", word):
            tokenized_word.extend([word])

        split_pred = self.camel_case_split(word)
        if len(split_pred) >= 2:
            tokenized_word = split_pred
            
        # BPE encoding
        else:
            encoded_p = self.bpemb_en.encode(word)
            encoded_p[0] = encoded_p[0].replace('▁', '')
            tokenized_word = encoded_p
        
        return tokenized_word


    def search_antonyms(self, word):
        antonyms = []

        # tokenized_word = self.word_tokenizer(word)
        for w in word:
            for syn in wordnet.synsets(w):
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
        
        return list(set(antonyms))

    def score(self, t):
        return t[1]


    def align_relations(self, relation_set):

        # Calculate similarity of relations
        marked_text = list()
        for text in relation_set:
            marked_text.append("[CLS] " + text + " [SEP]")

        tokenized_text = list()
        for text in marked_text:
            tokenized_text.append(self.tokenizer.tokenize(text))

        indexed_tokens = list()
        for token in tokenized_text:
            indexed_tokens.append(self.tokenizer.convert_tokens_to_ids(token))

        segments_ids = list()
        for indexed_t in indexed_tokens:
            segments_ids.append([1] * len(indexed_t))

        tokens_tensor = list()
        segments_tensors = list()
        for t in indexed_tokens:
            tokens_tensor.append(torch.tensor([t]))
            
        for s in segments_ids:
            segments_tensors.append(torch.tensor([s]))

        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        model.eval()


        hidden_states = list()
        cnt = 0
        with torch.no_grad():
            for output in zip(tokens_tensor, segments_tensors):
                hidden_states.append(model(output[0], output[1])[2])
                cnt += 1

        token_embeddings = list()
        for h in hidden_states:
            token_embeddings.append(torch.stack(h, dim=0))
            
        for i in range(len(token_embeddings)):
            token_embeddings[i] = torch.squeeze(token_embeddings[i], dim=1)
            
        for i in range(len(token_embeddings)):
            token_embeddings[i] = token_embeddings[i].permute(1,0,2)


        all_token_vecs = []
        for token_embedding in token_embeddings:
            token_vecs_sum = []
            
            for token in token_embedding:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec)

            all_token_vecs.append(token_vecs_sum)
            
        token_vecs = list()
        for i in range(len(all_token_vecs)):
            token_vecs.append(all_token_vecs[i][1].tolist())

        cosine_sim = pdist(token_vecs, metric = 'cosine')
        square_form = 1 - squareform(cosine_sim)

        pred_arr = np.array(list(relation_set))
        all_list = []

        rank = 3
        for sim_list in square_form:
            idx_sim_list = []
            
            for i, sim_score in enumerate(sim_list):
                idx_sim_list.append((i, sim_score))
            
            idx_sim_list.sort(key=self.score, reverse=True)
            
            rank_list = list()
            for idx, value in idx_sim_list:
                pred_val = (pred_arr[idx], value)
                rank_list.append(pred_val)
                    
            all_list.append(rank_list[:rank])

        return all_list

    def camel_case_split(self, identifier):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0).lower() for m in matches]

    def remove_antonyms(self, pred_rank_list):

        pred_list = []
        pred_anto_list = []
        idx = 0

        for ranks in pred_rank_list:
            tokenized_word = []
            antonyms_list = []
            
            for pred_val in ranks:
                if re.findall("[0-9]+", pred_val[0]):
                    tokenized_word.append([pred_val[0].lower()])
                    continue
                
                split_pred = self.camel_case_split(pred_val[0])
                if len(split_pred) >= 2:
                    tokenized_word.append(split_pred)
                
                else:
                    encoded_p = self.bpemb_en.encode(pred_val[0])
                    encoded_p[0] = encoded_p[0].replace('▁', '')
                    tokenized_word.append(encoded_p)

            pred_list.append(tokenized_word)

            for w in tokenized_word[0]:
                antonyms = []
                antonyms = self.search_antonyms(w)
                
                if antonyms:
                    antonyms_list.extend(antonyms)
            
            pred_anto_list.append(antonyms_list)
            
            idx += 1


        idx = 0
        for preds, antonyms in zip(pred_list, pred_anto_list):
            sim_preds = []
            
            if antonyms:
                for i in range(len(preds[1:])):
                    if len(preds[i+1]) >= 2:
                        for p in preds[i+1]:
                            if p in antonyms:
                                break
                            elif p not in antonyms and p != preds[i+1][-1]:
                                sim_preds.append(''.join(preds[i+1]))
                                sim_preds = list(set(sim_preds))
                                
                    elif len(preds[i+1]) < 2 and preds[i+1] in antonyms:
                        break
                        
                pred_list[idx] = [''.join(preds[0])] + sim_preds
            
            else:
                for i in range(len(preds[1:])):
                    sim_preds.append(''.join(preds[i+1]))
                    sim_preds = list(set(sim_preds))
                    
                pred_list[idx] = [''.join(preds[0])] + sim_preds

            idx += 1

        return pred_list



    def search(self, kg_triples, doc_triples):

        rel_dict = defaultdict(set)
        for s, p, o in kg_triples:
            rel_dict[p].add((s, o))

        for triple in doc_triples:
            rel_dict[triple["relation"]].add((triple["subject"], triple["object"]))

        print("Aligning relations ...")
        cosine_sim_rel_set = self.align_relations(rel_dict.keys())
        # print("cosine_sim_rel_set:", cosine_sim_rel_set)
        # print()

        print("Removing antonyms ...")
        sim_rel_collection = self.remove_antonyms(cosine_sim_rel_set)
        # print("sim_rel_collection:", sim_rel_collection)

        facts_set = set()
        for rel in rel_dict:
            for collection in sim_rel_collection:
                if rel in collection:
                    break
                else:
                    collection = []
            
            for sim_rel in collection:
                facts_set.add((list(list(rel_dict[rel])[0])[0], sim_rel, list(list(rel_dict[rel])[0])[1]))

        merged_supporting_facts = []
        for fact in facts_set:
            merged_supporting_facts.append((fact[0], fact[1], fact[2]))

        return merged_supporting_facts         # triple list: [("팀 밀러","데드풀","감독"), ("패트릭 휴스","킬러의 보디가드","감독"), ("패트릭 휴스","영화 감독","직업"),
                                               #               ("subject_1", "relation_1", "object_1"), ("subject_2", "relation_2", "object_2"), ("subject_3", "relation_3", "object_3")]



if __name__ == "__main__":
    example_content = {
        "question":{
            "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
            "language":"kr",
            "domain":"common-sense"
        },

        "kg_triples":[
            ["팀 밀러","데드풀","감독"],
            ["패트릭 휴스","킬러의 보디가드","감독"],
            ["패트릭 휴스","영화 감독","직업"]
        ],

        "doc_triples":[{
            "subject":"subject_1",
            "relation":"relation_1",
            "object":"object_1"
        },
        {
            "subject":"subject_2",
            "relation":"relation_2",
            "object":"object_2"
        },
        {
            "subject":"subject_3",
            "relation":"relation_3",
            "object":"object_3"
        }
        ]
    }

    import sys
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        example_content = json.load(f)

    
    model = Merging_supporting_graphs()
    result = model.do_search(example_content)

    print(result)
    for d in sorted(result, key=lambda x:x[0]):
        print(d)
