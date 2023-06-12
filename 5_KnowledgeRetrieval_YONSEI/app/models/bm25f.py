import subprocess
import json

class Retrieval_bm25f(object):
    def __init__(self):
        config = json.load(open("./data/bm25f/config_bm25f.json"))
        self.index_path = config['index']
        self.weights_path = config['weights']
        self.traversals_path = config['traversals']
        self.field_path = config['field']
        self.query_path = config['query']
        self.output_file = config['outputfile']

    def create_query_file(self, question_txt):
        with open(self.query_path, 'w') as f:
            line = 'Q_ID\t' + question_txt
            f.write(line)

    def retrieve(self):
        command = """galago batch-search batch-search --index={} \
            --queryFormat=tsv --operatorWrap=bm25f \
            --queries={} --outputFile={} {} {} {}""".format(self.index_path, self.query_path, self.output_file, 
                                                            self.weights_path, self.traversals_path, self.field_path)

        print("Execute galago BM25F command below ..")
        print(command)
        subprocess.call(command, shell=True)

        return self.output_file

    def extract_entities(self, output):
        
        ent_list = []
        with open(output) as f:
            for line in f:
                ent = line.split('\t')[2].split('resource/')[-1][:-1]
                ent_list.append(ent)
        
        return ent_list
