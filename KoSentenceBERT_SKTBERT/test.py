from sentence_transformers import SentenceTransformer, util
import numpy as np

model_path = './output/training_sts'

embedder = SentenceTransformer(model_path)

# Corpus with example sentences
corpus = ['당당한 목소리로',
          '서러운 목소리로',
          '의젓한 목소리로',
          '화가난 목소리로',
          '다급한 목소리로',
          '걱정하는 목소리로',
          '공손한 목소리로',
          '쌀쌀맞은 목소리로',
          '다정한 목소리로']

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['차갑게',
           '따뜻하게',
           '분하게']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
