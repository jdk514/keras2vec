
class Encoder():
    def __init__(self, items):
        """Take in items to encode
        Args:
            items (:obj:`list` of objects)"""
        self.encode(items)

    def encode(self, items):
        """Take in items to encode
        Args:
            items (:obj:`list` of objects)"""
        encoder = {}
        inverse_encoder = {}
        for ix, item in enumerate(items):
            encoder[item] = ix
            inverse_encoder[ix] = item

        self.encoder = encoder
        self.inverse_encoder = inverse_encoder

    def transform(self, item):
        """Encodes a given object
        Args:
            item (object): Object to encode
        Returns:
            int: integer encoding of the item
        """

        try:
            encoded_item = self.encoder[item]
        except KeyError:
            raise KeyError("Provided item does not have an encoding.")

        return encoded_item

    def inverse_transform(self, index):
        """Reverses the encoding for a given index
        Args:
            int: index to reverse encoding
        Returns:
            object: decoded object"""

        try:
            item = self.inverse_encoder[index]
        except:
            raise KeyError("Provided encoding does not exist.")

        return item



class Keras2Vec_Generator():

    def __init__(self, documents):
        """Prepare a generator for the provided documents, enabling
        Keras2Vec to generate embeddings

        Args:
            documents (:obj:`list` of :obj:`Document`): List of documents to vectorize
        """
        self.doc_vocab, self.label_vocab, self.text_vocab = None

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

