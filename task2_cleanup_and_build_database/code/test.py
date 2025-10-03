#%%
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

#%%
pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)


#%%


from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
sentence1 = "你好我是狗"
sentence2 = "他是猫"
sentence3 = "国王下令"
model = FlagModel('BAAI/bge-large-zh-v1.5', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1, convert_to_tensor=True)
embeddings_2 = model.encode(sentences_2)

embeddings1 = model.encode(sentence1)
embeddings2 = model.encode(sentence2)
embeddings3 = model.encode(sentence3)

embeddings1 @ embeddings2
embeddings2 @ embeddings3
embeddings1 @ embeddings3

embeddings_1.shape
similarity = embeddings_1 @ embeddings_2.T
print(similarity)


input_file = "/Volumes/t7/projects/xpf/task_cluster_column_headers/out/column_headers_only.txt"
corpus_sentences = set()

with open(input_file, encoding="utf8") as f:
    for line in f:
        corpus_sentences.add(line.strip())

corpus_sentences = list(corpus_sentences)
corpus_embeddings = model.encode(corpus_sentences)


# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T