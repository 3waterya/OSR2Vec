import networkx
import obonet
import numpy as np
import joblib
graph = obonet.read_obo('./go.obo')

id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}

sentence=[]
for id_, data in graph.nodes(data=True):
        n = data['name']
        node = name_to_id[n]
        for child, parent, key in graph.out_edges(node, keys=True):
            sentence.append ( id_to_name[child]+ ' '+key+ ' '+id_to_name[parent] )
        for parent, child, key in graph.in_edges(node, keys=True):
            sentence.append ( id_to_name[parent]+ ' '+key+ ' '+id_to_name[child] )