import random

import numpy as np

from keras2vec.encoder import Encoder

# TODO: Implement as a keras.utils.Sequence class
class DataGenerator:
    """The DataGenerator class is used to encode documents and generate training/testing
    data for a Keras2Vec instance. Currently this object is only used internally within the
    Keras2Vec class and not intended for direct use.

    Args:
        documents (:obj:`list` of :obj:`Document`): List of documents to vectorize
    """

    def __init__(self, documents, seq_size, neg_samples=5):
        self.doc_vocab = self.label_vocab = self.text_vocab = None
        self.doc_enc = self.label_enc = self.text_enc = None

        self.neg_samples = neg_samples
        self.seq_size = seq_size

        # TODO: Change the documents attribute to encoded documents
        [doc.gen_windows(seq_size) for doc in documents]
        self.documents = documents
        self.build_vocabs()
        self.create_encodings()

    def build_vocabs(self):
        """Build the vocabularies for the document ids, labels, and text of
        the provided documents"""

        doc_vocab = set()
        label_vocab = set()
        text_vocab = set([''])

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

    # TODO: Enable partial generation of data
    def generator(self):
        """Generates a single epoch of encoded data for the keras model"""
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
                batch_labels.append(np.array(enc_labels))
                batch_words.extend(enc_words)
                batch_outputs.append(outputs)

            if len(self.label_vocab) > 0:
                inputs = [np.array(batch_docs),
                          np.vstack(batch_labels),
                          np.vstack(batch_words)]
            else:
                inputs = [np.vstack(batch_docs), np.vstack(batch_words)]

            outputs = np.vstack(batch_outputs)
            yield [np.vstack(batch_docs), np.array(batch_words)], outputs


    # TODO: Replace with generator
    def neg_sampling(self, window):
        neg_samples = []
        win_ix = int((self.seq_size-1)/2)
        center_word = window[win_ix]
        word_dict = self.text_vocab.copy()
        word_dict.remove(center_word)
        dict_len = len(word_dict)

        for ix in range(self.neg_samples):
            if len(word_dict) < 1:
                break
            rep_word = random.sample(word_dict, 1)[0]
            word_dict.remove(rep_word)
            new_win = window.copy()
            new_win[win_ix] = rep_word
            neg_samples.append(new_win)

        return neg_samples


    # TODO: incorporate neg_sampling
    def encode_doc(self, doc, neg_sampling=False, num_neg_samps=5):
        """Encodes a document for the keras model

        Args:
            doc(Document): The document to encode
            neg_sampling(Boolean): Whether or not to generate negative samples for the document
            **NOTE**: Currently not implemented"""
        docs = []
        labels = []
        words = []
        outputs = []

        enc_doc = self.doc_enc.transform(doc.doc_id)
        enc_labels = [self.label_enc.transform(lbl) for lbl in doc.labels]
        for window in doc.windows:
            enc_words = [self.text_enc.transform(word) for word in window]
            docs.append(enc_doc)
            labels.append(enc_labels)
            words.append(enc_words)
            outputs.append(1)

            if self.neg_samples > 0:
                for neg_samp in self.neg_sampling(window):
                    enc_words = [self.text_enc.transform(word) for word in neg_samp]
                    docs.append(enc_doc)
                    labels.append(enc_labels)
                    words.append(enc_words)
                    outputs.append(0)


        ret = (np.vstack(docs),
               np.vstack(labels),
               words,
               np.vstack(outputs))

        return ret
