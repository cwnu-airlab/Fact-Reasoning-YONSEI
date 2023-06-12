import json
import os
from models.bm25f import Retrieval_bm25f
from models.lda import Retrieval_lda
from models.w2v import Retrieval_w2v
from collections import defaultdict

class Service:
    task = [
        {
            'name': 'Knowledge_Retrieval', 
            'description':'Retrieve supporting knwoledge from knowledge graph over a natural language question'
        }
    ]

    def __init__(self):
        self.graph_retrieval = Supporting_graph_retrieval()

    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            ret = self.graph_retrieval.do_search(content)

            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400

class Supporting_graph_retrieval(object):
    def __init__(self):
        pass

    def do_search(self, content):
        question = content.get('question', None)
        graph_path = content.get('knowledge_graph_path', None)
        if question is None:
            return {
                'error': "invalid question"
            }
        elif not os.path.exists(graph_path):
            return {
                'error': "knowledge graph file doesn't exist"
            }
        else:
            question_txt = question.get('text', '')
            # question_ln = question.get('language', 'en')
            # question_domain = question.get('domain', 'common-sense')

            if question_txt == '':
                return {
                    'error': "invalid question text"
                }
            else:
                return self.search(question_txt, graph_path)


    def next_hop_ent_set(self, graph, start_ent_set):
        next_ent_set_ = set()
        
        for ent in start_ent_set:
            if ent in graph:
                for pred_obj in graph[ent]:
                    next_ent_set_.add(pred_obj[1])
                
        return next_ent_set_


    def extract_neighborhood(self, ents, graph_path):
        print("ents:", ents)
        graph_dict = defaultdict(list)

        with open(graph_path, encoding='utf-8') as f:
            graph = set(line.rstrip('\n') for line in f)
        
        print("Creating graph ...")
        for line in graph:
            subj, pred, obj = line.split(maxsplit=2)
            subj = subj[1:-1]
            pred = pred[1:-1]
            obj = obj[1:-1]
            
            if [pred, obj] in graph_dict[subj]:
                continue
            else:
                graph_dict[subj].append([pred, obj])

        selected_ent_set = set(ents)
        next_ent_set = set(ents)

        n_hop = 1
        for n in range(n_hop):
            print('%s-hop\n' % (n+1))
            new_next_ent_set = set()
            
            next_ent_set = self.next_hop_ent_set(graph_dict, next_ent_set)
            
            print('num of next %s hop ents:' % (n+1), f'{len(next_ent_set):,}')
            for ent in next_ent_set:
                if ent not in selected_ent_set:
                    new_next_ent_set.add(ent)

            print('num of new next ents to search:', f'{len(new_next_ent_set):,}')
            selected_ent_set = selected_ent_set.union(next_ent_set)
            next_ent_set = new_next_ent_set
            
            if not new_next_ent_set or (n == n_hop - 1):
                break


        n_hop_graph = {
            "kg_triples":[]
            }
        for ent in selected_ent_set:
            if ent in graph_dict:
                for pred_obj in graph_dict[ent]:
                    n_hop_graph["kg_triples"].append([ent, pred_obj[0], pred_obj[1]])


        return n_hop_graph



    def search(self, question_txt, graph_path):

        ### question_txt, graph_path : input on bm25f, lda, w2v
        retrieval_bm25f = Retrieval_bm25f()
        retrieval_lda = Retrieval_lda()
        retrieval_w2v = Retrieval_w2v()
        #retrieval_w2v = Retrieval_w2v(question_txt)

        ### BM25F
        retrieval_bm25f.create_query_file(question_txt)
        bm25f_output_file = retrieval_bm25f.retrieve()
#         bm25f_output_file = retrieval_bm25f.retrieve(question_txt)
        ent_list_bm25f = retrieval_bm25f.extract_entities(bm25f_output_file) # list("A", "B", "C", ...)

        ### LDA
        ent_list_lda = retrieval_lda.retrieve(question_txt, ent_list_bm25f) # list("A", "B", "C", ...)

        ### Interpolate with W2V
        output_ents = retrieval_w2v.retrieve(question_txt, ent_list_lda)

        ### extract 2-hop neighborhoods from retrieved entities
        supporting_facts_from_kg = self.extract_neighborhood(output_ents, graph_path)
    
        return supporting_facts_from_kg                 # content: {
                                                        #               "triples": [["팀 밀러","데드풀","감독"], ["패트릭 휴스","킬러의 보디가드","감독"], ["패트릭 휴스","영화 감독","직업"]]
                                                        #          }



if __name__ == "__main__":

    example_content = {
        "question":{
            "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
            "language":"kr",
            "domain":"common-sense"
        },

        "knowledge_graph_path": './data/graph_processed.tsv'
    }
    model = Supporting_graph_retrieval()
    result = model.do_search(example_content)
    print(result)
