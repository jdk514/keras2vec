from keras2vec.keras2vec import Keras2Vec
from keras2vec.document import Document

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def layer_dist(embeddings, id_1, id_2):
    doc1 = embeddings[id_1 - 1].reshape(1, -1)
    doc2 = embeddings[id_2 - 1].reshape(1, -1)
    return cosine_similarity(doc1, doc2)


if __name__ == "__main__":

    docs = [Document(1, [], "Test Document 01"),
            Document(1, [], "Test Document 02"),
            Document(1, [], "Test Document 03"),
            Document(1, [], "Test Document 04"),
            Document(2, [], "Test Document 05"),
            Document(2, [], "Test Document 06"),
            Document(2, [], "Test Document 07"),
            Document(2, [], "Test Document 08"),
            Document(3, [], "Random words to experiment"),
            Document(3, [], "Random words to experiment"),
            Document(3, [], "Random words to experiment"),
            Document(3, [], "Random words to experiment"),
            Document(4, [], "Random text to experiment"),
            Document(4, [], "Random text to experiment"),
            Document(4, [], "Random text to experiment"),
            Document(4, [], "Random text to experiment")
            ]


    doc2vec = Keras2Vec(docs, embedding_size=16, seq_size=1)
    doc2vec.build_model()
    doc2vec.fit(15)

    # TODO: Add method to pull doc/word/label embeddings
    embeddings = doc2vec.train_model.get_layer('doc_embedding').get_weights()[0]
    print(layer_dist(embeddings, 1, 4))