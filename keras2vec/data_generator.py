from keras2vec.encoder import Encoder


class DataGenerator:

    def __init__(self, documents):
        """Prepare a generator for the provided documents, enabling
        Keras2Vec to generate embeddings

        Args:
            documents (:obj:`list` of :obj:`Document`): List of documents to vectorize
        """
        self.doc_vocab = self.label_vocab = self.text_vocab = None
        self.doc_enc = self.label_enc = self.text_enc = None

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
        pass

