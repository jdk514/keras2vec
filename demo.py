import numpy as np

from keras2vec.keras2vec import Keras2Vec
from keras2vec.document import Document

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def doc_similarity(embeddings, id_1, id_2):
    doc1 = embeddings[id_1].reshape(1, -1)
    doc2 = embeddings[id_2].reshape(1, -1)
    return cosine_similarity(doc1, doc2)[0][0] # , euclidean_distances(doc1, doc2)


if __name__ == "__main__":

    color_docs = ["red yellow green blue orange violet green blue orange violet",
                   "blue orange green gray black teal tan blue violet gray black teal",
                   "blue violet gray black teal yellow orange tan white brown",
                   "black blue yellow orange tan white brown white green teal pink blue",
                   "orange pink blue white yellow black black teal tan",
                   "white green teal gray black pink blue blue violet gray black teal yellow",
                   ]

    animal_docs = ["cat dog rat gerbil hamster goat lamb goat cow rat dog pig",
                   "lamb goat cow rat dog pig dog chicken goat cat cow pig",
                   "pig lamb goat rat gerbil dog cat dog rat gerbil hamster goat",
                   "dog chicken goat cat cow pig gerbil goat cow pig gerbil lamb",
                   "rat hamster pig dog chicken cat lamb goat cow rat dog pig dog",
                   "gerbil goat cow pig gerbil lamb rat hamster pig dog chicken cat"
                   ]
    shape_docs = ["square triangle hexagon circle octagon cube",
                  "pyramid circle cube pentagon cylinder trapezoid",
                  "diamond octagon quadrilateral cylinder rectangle square",
                  "trapezoid cube hexagon diamond triangle circle cylinder",
                  "square rectangle quadrilateral octagon pentagon square"]

    animal_color_docs = ['goat green rat gerbil yellow dog cat blue white',
                         'gerbil black pink blue lamb rat hamster gray pig dog',
                         'orange pink cat cow pig black teal gerbil tan',
                         'hamster pig orange violet dog chicken orange tan']

    inference_doc = "red yellow green blue orange violet green blue orange violet"

    doc_count = 0
    keras_docs = []

    keras_docs.extend([Document(doc_count+ix, text, ['color']) for ix, text in enumerate(color_docs)])
    doc_count = len(keras_docs)
    keras_docs.extend([Document(doc_count+ix, text, ['animal']) for ix, text in enumerate(animal_docs)])
    doc_count = len(keras_docs)
    keras_docs.extend([Document(doc_count + ix, text, ['shape']) for ix, text in enumerate(shape_docs)])
    doc_count = len(keras_docs)
    keras_docs.extend([Document(doc_count + ix, text, ['animal', 'color']) for ix, text in enumerate(animal_color_docs)])

    # TODO: Add ability to auto-select embedding and seq_size based on data
    doc2vec = Keras2Vec(keras_docs, embedding_size=24, seq_size=1)
    doc2vec.build_model()
    # If the number of epochs is to low, the check at the bottom may fail!
    print("Training Model:")
    history = doc2vec.fit(250, verbose=1)
    print("\ttraining complete!")

    embeddings = doc2vec.get_doc_embeddings()

    print("Beginning Evaluation:")
    """Docs 0-5 are colors while 6-11 are animals. The cosine distances for
    docs from the same topic (colors/animals) should approach 1, while
    disimilar docs, coming from different topics, should approach -1"""
    if doc_similarity(embeddings, 2, 4) > doc_similarity(embeddings, 1, 10):
        print("\t- Like topics are more similar!")
    else:
        print("\t- Something went wrong during training.")

    """Using the trained model we can now infer document vectors by training
    against a model where only the inference layer is trainable"""

    doc2vec.infer_vector(Document(0, inference_doc, ['color']), lr=.1, epochs=50)
    infer_vec = doc2vec.get_infer_embedding()
    infer_dist = cosine_similarity(infer_vec.reshape(1, -1), embeddings[0].reshape(1, -1))[0][0]
    infer_dist = "{0:0.2f}".format(infer_dist)
    print(f'\t- Document 0 has a cosine similarity of {infer_dist} between train and inferred vectors')


    """Label analysis: shape should be farther away than animal and color"""
    label_embeddings = doc2vec.get_label_embeddings()
    shape_vector = doc2vec.get_label_embedding('shape').reshape(1, -1)
    animal_vector = doc2vec.get_label_embedding('animal').reshape(1, -1)
    color_vector = doc2vec.get_label_embedding('color').reshape(1, -1)
    animal_color_dist = cosine_similarity(animal_vector, color_vector)[0][0]
    shape_color_dist = cosine_similarity(shape_vector, color_vector)[0][0]
    if animal_color_dist > shape_color_dist:
        print("\t- Label distances look good!")
    else:
        print("\t- Something went wrong with the labels.")