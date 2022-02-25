from sentence_transformers import SentenceTransformer, SentencesDataset, losses, InputExample,models
from torch import nn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

obo_file = 'pygosemsim/_resources/go-basic.obo'

fp=open(obo_file,'r')
obo_txt=fp.read()
fp.close()
#取[Term]和[Typedef]中间的字符信息
obo_txt=obo_txt[obo_txt.find("[Term]")-1:]
obo_txt=obo_txt[:obo_txt.find("[Typedef]")]
# obo_dict=parse_obo_txt(obo_txt)
id_namespace_dicts = {}
id_name_dicts = {}
id_def_dicts = {}
id_namedef_dicts = {}

for Term_txt in obo_txt.split("[Term]\n"):
    if not Term_txt.strip():
        continue
    name = ''
    ids = []
    for line in Term_txt.splitlines():
        if   line.startswith("id: "):
            ids.append(line[len("id: "):]) 
        elif line.startswith("name: "):
            name=line[len("name: "):]
        elif line.startswith("namespace: "):
             name_space=line[len("namespace: "):]
        elif line.startswith("alt_id: "):
            ids.append(line[len("alt_id: "):])
        elif line.startswith("def: "):
            defi=(line[len("def: "):])
            flag = defi.find('."') 
            defi = defi[1:flag-1]
    
    for t_id in ids:
        id_namedef_dicts[t_id] = name +" "+defi
        
def extrtactembeddingbysbert( sbert, num_proj_hidden=256):
    prot_emb_dict = {}
    for key,value in tqdm( id_namedef_dicts.items()):
        feature = sbert.encode([id_namedef_dicts[key]], convert_to_tensor=True)    
        prot_emb_dict[key] = feature.cpu().numpy()
    return prot_emb_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
sbert_ = SentenceTransformer('output/OSR2vec' , device=device)
sbert_embedding_wo_train = extrtactembeddingbysbert(sbert_,num_proj_hidden=256)

joblib.dump(sbert_embedding_wo_train, filename='output/OSR2vec_emb')