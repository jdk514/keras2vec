import random

from keras2vec.encoder import Encoder

# TODO: Implement as a keras.utils.Sequence class
class DataGenerator:

    def __init__(self, documents):
        """Prepare a generator for the provided documents, enabling
        Keras2Vec to generate embeddings

        Args:
            documents (:obj:`list` of :obj:`Document`): List of documents to vectorize
        """
        self.doc_vocab = self.label_vocab = self.text_vocab = None
        self.doc_enc = self.label_enc = self.text_enc = None

        # TODO: Change the documents attribute to encoded documents
        self.documents = documents
        self.build_vocabs()
        self.create_encodings()

    def build_vocabs(self):
        """Build the vocabularies for the document ids, labels, and text of
        the provided documents"""

        doc_vocab = set()
        label_vocab = set()
        text_vocab = set()

        for doc in self.documents:
            doc_vocab.add(doc.doc_id)
            label_vocab.update(doc.labels)
            text_vocab.update(doc.text)

        self.doc_vocab = doc_vocab
        self.label_vocab = label_vocab
        self.text_vocab = text_vocab

    def create_encodings(self):
        """Build the encodings for each of the provided data types"""
        self.doc_enc = Encoder(self.doc_vocab)
        self.label_enc = Encoder(self.label_vocab)
        self.text_enc = Encoder(self.text_vocab)

    def generator(self):
        indicies = list(range(len(self.documents)))

        while True:
            random.shuffle(indicies)

            batch_docs = []
            batch_labels = []
            batch_words = []
            batch_outputs = []
            for ix in indicies:
                curr_doc = self.documents[ix]
                enc_doc, enc_labels, enc_words, outputs = self.encode_doc(curr_doc)
                batch_docs.append(enc_doc)
                batch_labels.append(enc_labels)
                batch_words.append(enc_words)
                batch_outputs.append(outputs)


            yield (batch_docs, batch_labels, batch_words, batch_outputs)


    # TODO: incorporate neg_sampling
    def encode_doc(self, doc):
        enc_doc = self.doc_enc(doc.doc_id)
        enc_labels = [self.label_enc(lbl) for lbl in doc.labels]
        enc_words = [self.text_enc(word) for word in doc.text]
        outputs = 1

        return enc_doc, enc_labels, enc_words, outputs
