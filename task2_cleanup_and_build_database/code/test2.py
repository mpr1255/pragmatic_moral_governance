#%%
import pandas as pd
import tiktoken
from openai import OpenAI
client = OpenAI()
import numpy as np

model="text-embedding-3-small"
sentence1 = "你好我是狗"
sentence2 = "他是猫"
sentence3 = "国王下令"

embed1   = np.array(client.embeddings.create(input = sentence1, model=model).data[0].embedding)
embed2 = np.array(client.embeddings.create(input = sentence2, model=model).data[0].embedding)
embed3 = np.array(client.embeddings.create(input = sentence3, model=model).data[0].embedding)
#%%

from numba import jit
import numpy as np

@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


cosine_similarity_numba(embed1, embed2)
cosine_similarity_numba(embed2, embed3)
cosine_similarity_numba(embed1, embed3)