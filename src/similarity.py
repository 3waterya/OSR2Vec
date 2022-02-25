from pygosemsim import graph,download,term_set,similarity,annotation
import pandas as pd
import networkx as nx
import functools
import numpy as np
import multiprocessing as mp
import evaluating as evlt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from IPython import display
import joblib


G = graph.from_resource("go-basic")
similarity.precalc_lower_bounds(G)
#不同数据集
dataset = 'PPI_DM1'
if 'HS' in dataset:
    annot = annotation.from_resource("goa_human")
    dataset_file_path = 'data/kgsim/data_set/'+dataset+'.csv'
    print('HS')
elif 'EC' in dataset:
    annot = annotation.from_resource("ecocyc")
    dataset_file_path = 'data/kgsim/data_set/'+dataset+'.csv'
    print('EC')
elif 'SC' in dataset:
    annot = annotation.from_resource("goa_yeast")
    dataset_file_path = 'data/kgsim/data_set/'+dataset+'.csv'
    print('SC')
elif 'DM' in dataset:
    annot = annotation.from_resource("goa_fly")
    dataset_file_path = 'data/kgsim/data_set/'+dataset+'.csv'
    print('DM')

#proteins集合相似度
def compute_bma_results(dataset_file_path, annot, method):
    compute_results = []
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    for index, row in tqdm(measure_df.iterrows()):
        p1 = row['Uniprot ID1']
        p2 = row['Uniprot ID2']
        if p1 not in annot.keys() or p2 not in annot.keys():
            compute_results.append(0)
        else:
            trpv1 = annot[p1]["annotation"].keys()
            trpa1 = annot[p2]["annotation"].keys()
            sf = functools.partial(term_set.sim_func, G, method)
            compute_results.append(term_set.sim_bma(trpv1, trpa1, sf))
    return compute_results

#定义BMA
def BMA(sent1,sent2):
    NAN_VALUE = float('nan')
    summation_set12 = 0.0
    summation_set21 = 0.0
    for id1 in sent1:
        similarity_values = []
        for id2 in sent2:
            similarity_values.append(cosine_similarity(id1, id2)[0][0])
        summation_set12 += max(similarity_values + [NAN_VALUE])
    for id2 in sent2:
        similarity_values = []
        for id1 in sent1:
            similarity_values.append(cosine_similarity(id1, id2)[0][0])
        summation_set21 += max(similarity_values + [NAN_VALUE])
    if (len(sent1) + len(sent1)) == 0:
        bma = 0
    else:
        bma = (summation_set12 + summation_set21) / (len(sent1) + len(sent2))
    return bma

#embbeding
def extrtactembeddingbysbert(dataset_file_path,anno_dict, all_go_emb, num_proj_hidden=256):
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    all_protein = list(measure_df['Uniprot ID1']) + list(measure_df['Uniprot ID2'])
    all_protein = list(set(all_protein))
    prot_emb_dict = {}
    for prot_id in tqdm(all_protein):
        embedding = []
        if prot_id in anno_dict.keys():
            all_gos = list(anno_dict[prot_id]["annotation"].keys())
            for go in all_gos:
                embedding.append(all_go_emb[go]  )
            if embedding == []:
                embedding.append( np.zeros((1, num_proj_hidden)))
            prot_emb_dict[prot_id] = embedding
        else:
            embedding.append( np.zeros((1, num_proj_hidden)))
            prot_emb_dict[prot_id] = embedding
    return prot_emb_dict

#similarity scores
def sbert_bma_score(dataset_file_path,anno_dict,prot_emb_dict, num_proj_hidden=256):
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    bert_scores = []
    pool = mp.Pool(mp.cpu_count())
    bert_scores = pool.starmap(BMA, [(prot_emb_dict[row['Uniprot ID1']], prot_emb_dict[row['Uniprot ID2']]) for index, row in measure_df.iterrows()])
    pool.close() 
    return bert_scores

#OSR2vec scores
all_go_emb = joblib.load('output/OSR2vec_emb')
prot_emb_dict = extrtactembeddingbysbert(dataset_file_path,annot, all_go_emb,num_proj_hidden=768)
OSR2vecscores = sbert_bma_score(dataset_file_path,annot, prot_emb_dict,num_proj_hidden=768)

#other methods
lin_bma_results= compute_bma_results(dataset_file_path, annot, similarity.lin)
wang_bma_results = compute_bma_results(dataset_file_path, annot, similarity.wang)

#pearson coorelation 
dataset_type = 'PPI'
test_dataset = dataset_file_path
benchmark_dataset = dataset_file_path
measures = {
            'OSR2vecscores':np.array(OSR2vecscores),
            'bma_lin':np.array(lin_bma_results), 
            'bma_wang': np.array(wang_bma_results)     
}
results = evlt.correlation_calculation(benchmark_dataset, measures, dataset_type)
# display.display(results)
