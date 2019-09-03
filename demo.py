from keras2vec.keras2vec import Keras2Vec
from keras2vec.document import Document

if __name__ == "__main__":

    docs = [Document(1, [], "Test Document 01"),
            Document(1, [], "Test Document 02"),
            Document(1, [], "Test Document 03"),
            Document(1, [], "Random words to experiment"),
            Document(1, [], "Random words to experiment"),
            Document(1, [], "Random words to experiment")
            ]


    doc2vec = Keras2Vec(docs)
    doc2vec.build_model()
    doc2vec.fit(5)