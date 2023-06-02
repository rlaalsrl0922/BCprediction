import torch.nn.functional as F
import torch
import pandas as pd
from collections import Counter

#input = torch.LongTensor([[1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4],
#                          [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4],
#                          [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4],
#                          [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4], [1], [4],])
#embe = torch.nn.Embedding(4874,5)

#print(embe(input))

annotation=pd.read_csv('/data/minjae/BC/swbd_equal.tsv',delimiter='\t',encoding='utf-8')
#print(len(Counter(annotation['folder'])))
a = []
for i in range(len(annotation)):
    if annotation.iloc[i,6] not in a:
        a.append(annotation.iloc[i,6])

print(len(a))         

