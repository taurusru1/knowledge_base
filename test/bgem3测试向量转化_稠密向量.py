import time

from utils.embedding_utils import get_bge_m3_ef

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
model = get_bge_m3_ef()

start_time = time.time()
docs_embeddings = model.encode_documents(docs)

dense = docs_embeddings["dense"]


# for emb in dense:
#     print(type(emb.tolist()))

dense_list = [emb.tolist() for emb in dense]
print(dense_list)
end_time = time.time()
print("执行用时：", end_time - start_time)
