# Keras2Vec
A Keras implementation, enabling gpu support, of Doc2Vec

## Using Keras2Vec
This package can be installed via pip:
        
        pip install keras2vec

Documentation for Keras2Vec can be found on [readthedocs](https://keras2vec.readthedocs.io/en/latest/).

## Example Usage
```python
from keras2vec.keras2vec import Keras2Vec
from keras2vec.document import Document

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def doc_similarity(embeddings, id_1, id_2):
    doc1 = embeddings[id_1].reshape(1, -1)
    doc2 = embeddings[id_2].reshape(1, -1)
    return cosine_similarity(doc1, doc2)[0][0] # , euclidean_distances(doc1, doc2)


docs =["red yellow green blue orange violet green blue orange violet",
       "blue orange green gray black teal tan blue violet gray black teal",
       "blue violet gray black teal yellow orange tan white brown",
       "black blue yellow orange tan white brown white green teal pink blue",
       "orange pink blue white yellow black black teal tan",
       "white green teal gray black pink blue blue violet gray black teal yellow",
       "cat dog rat gerbil hamster goat lamb goat cow rat dog pig",
       "lamb goat cow rat dog pig dog chicken goat cat cow pig",
       "pig lamb goat rat gerbil dog cat dog rat gerbil hamster goat",
       "dog chicken goat cat cow pig gerbil goat cow pig gerbil lamb",
       "rat hamster pig dog chicken cat lamb goat cow rat dog pig dog",
       "gerbil goat cow pig gerbil lamb rat hamster pig dog chicken cat"
       ]

keras_docs = [Document(ix, [], doc) for ix, doc in enumerate(docs)]

doc2vec = Keras2Vec(keras_docs, embedding_size=24, seq_size=1)
doc2vec.build_model()
# If the number of epochs is to low, the check at the bottom may fail!
doc2vec.fit(25)

embeddings = doc2vec.get_doc_embeddings()

"""Docs 0-5 are colors while 6-11 are animals. The cosine distances for
docs from the same topic (colors/animals) should approach 1, while
disimilar docs, coming from different topics, should approach -1"""
if doc_similarity(embeddings, 2, 4) > doc_similarity(embeddings, 2, 10):
    print("Like topics are more similar")
else:
    print("Something went wrong during training!")
```

## Changelog
**Version 0.0.2:**
 - Added **get_doc_embeddings()**, **get_doc_embedding(doc)**, **get_word_embeddings()**, and **get_word_embedding(word)** so embeddings can be grabbed directly
 - Incorporated Neg-Sampling into Doc2Vec implementation
   - *Note: Neg-Sampling is now a parameter when instantiatng a Keras2Vec object*
 - Updated Doc2Vec model
   - Concatenating document embedding to the document's context, rather than averaging
   - Added a dense layer between concatenated layer and sigmoid output in attempt to improve performance
   - Updated optimizer to leverage Adamax rather than SGD in attempt to improve performance

**Version 0.0.1:**
 - Initial Release
 - Keras implementation of Doc2Vec