import scanpy as sc, anndata
import pandas as pd, time
import numpy as np
import torch
import esm
import gget

model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()

batch_converter = alphabet.get_batch_converter()
model.eval()

df = pd.read_csv("uniprot-proteins.tsv", sep="\t")

data = []

for i in range(5):
    try:
        seq_list = gget.seq(df["To"][i], translate=True)
    except Exception as e:
        print(e)

    if (len(seq_list) > 0):
        data.append(("protein" + str(i+1), seq_list[1]))

batch_labels, batch_strs, batch_tokens = batch_converter(data)

print(batch_labels)
print(batch_strs)
print(batch_tokens)