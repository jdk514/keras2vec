# Keras2Vec
A Keras implementation, with gpu support, of the Doc2Vec network

## Using Keras2Vec
-----
This package can be installed via pip:
        
        pip install keras2vec

Documentation for Keras2Vec can be found on [readthedocs](https://keras2vec.readthedocs.io/en/latest/).

## Example Usage
-----
```python
from keras2vec.keras2vec import Keras2Vec
from keras2vec.document import Document

from sklearn.metrics.pairwise import cosine_similarity

docs = [Document(1, [], "Test Document 01"),
        Document(1, [], "Test Document 02"),
        Document(1, [], "Test Document 03"),
        Document(1, [], "Test Document 04"),
        Document(2, [], "Random words to experiment"),
        Document(2, [], "Random words to experiment"),
        Document(2, [], "Random words to experiment"),
        Document(2, [], "Random words to experiment"),
        ]

doc2vec = Keras2Vec(docs)
doc2vec.build_model()
doc2vec.fit(5)

embeddings = doc2vec.train_model.get_layer('doc_embedding').get_weights()[0]
doc1 = embeddings[0].reshape(1, -1)
doc2 = embeddings[0].reshape(1, -1)
cosine_similarity(doc1, doc2)
```