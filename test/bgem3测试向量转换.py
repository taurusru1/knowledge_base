from utils.embedding_utils import get_bge_m3_ef

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
model = get_bge_m3_ef()
docs_embeddings = model.encode_documents(docs)

print("Embeddings:", docs_embeddings)
print("Dense document dim:", get_bge_m3_ef.dim["dense"], docs_embeddings["dense"][0].shape)
print("Sparse document dim:", get_bge_m3_ef.dim["sparse"], list(docs_embeddings["sparse"])[0].shape)
