import tensorflow as tf

from keras.layers import Input, Embedding, Average, Dense, Lambda, Concatenate, Flatten, Dot
from keras.optimizers import Adamax, Adam, SGD
from keras.models import Model

from keras2vec.data_generator import DataGenerator


# TODO: Fix naming convention between words, text, labels, docs
class Keras2Vec:
    """The Keras2Vec class is where the Doc2Vec model will be trained. By taking in a set of
    Documents it can begin to train against them to learn the embedding space that best represents
    the provided documents.

    Args:
        documents (:obj:`list` of :obj:`Document`): List of documents to vectorize
    """

    def __init__(self, documents, embedding_size=16, seq_size=3, neg_sampling=5):
        self.train_model = self.infer_model = None
        self.word_embeddings = self.doc_embeddings = None

        self.generator = DataGenerator(documents, seq_size, neg_sampling)
        # TODO: Fix method for getting model attributes
        self.doc_vocab = len(self.generator.doc_vocab)
        self.label_vocab = len(self.generator.label_vocab)
        self.word_vocab = len(self.generator.text_vocab)
        self.num_labels = len(documents[0].labels)

        self.embedding_size = embedding_size
        self.seq_size = seq_size


    def build_model(self):
        """Build both the training and inference models for Doc2Vec"""

        doc_ids = Input(shape=(1,))
        sequence = Input(shape=(self.seq_size,))
        if self.num_labels > 0:
            labels = Input(shape=(self.num_labels,))

            label_embedding = Embedding(input_dim=self.label_vocab,
                                        output_dim=self.embedding_size,
                                        input_length=self.num_labels,
                                        name="label_embedding")(labels)

            if self.num_labels > 1:
                split_label = Lambda(self.split_layer())(label_embedding)
                avg_labels = Average()(split_label)
            else:
                avg_labels = label_embedding

        doc_inference = Embedding(input_dim=1,
                                  output_dim=self.embedding_size,
                                  input_length=1,
                                  name='inferred_vector')(doc_ids)

        doc_embedding = Embedding(input_dim=self.doc_vocab,
                                  output_dim=self.embedding_size,
                                  input_length=1,
                                  name="doc_embedding")(doc_ids)

        seq_embedding = Embedding(input_dim=self.word_vocab,
                                  output_dim=self.embedding_size,
                                  input_length=self.seq_size,
                                  name="word_embedding")(sequence)

        if self.seq_size > 1:
            split_seq = Lambda(self.__split_layer())(seq_embedding)
            avg_seq = Average()(split_seq)
        else:
            avg_seq = seq_embedding

        if self.num_labels > 0:
            context = Average()([avg_labels, avg_seq])
        else:
            context = avg_seq

        # Build training model
        # train_merged = Dot(axes=-1, normalize=True)([doc_embedding, context])
        train_merged = Concatenate()([doc_embedding, context])
        train_flattened = Flatten()(train_merged)
        train_hidden = Dense(self.embedding_size * 3, activation='relu')(train_flattened)
        train_output = Dense(1, activation='sigmoid')(train_hidden)

        if self.num_labels > 0:
            train_model = Model(inputs=[doc_ids, labels, sequence],
                                outputs=train_output)
        else:
            train_model = Model(inputs=[doc_ids, sequence],
                                outputs=train_output)

        optimizer = Adamax(lr=0.1)
        train_model.compile(optimizer=optimizer,
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
        self.train_model = train_model


        # Build model for inference
        infer_merged = Average()([doc_inference, context])
        infer_flattened = Flatten()(infer_merged)
        infer_output = Dense(1, activation='sigmoid')(infer_flattened)

        if self.num_labels > 0:
            infer_model = Model(inputs=[doc_ids, labels, sequence],
                            outputs=infer_output)
        else:
            infer_model = Model(inputs=[doc_ids, sequence], outputs=infer_output)

        infer_model.compile(optimizer='adamax',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
        self.infer_model = infer_model


    def __split_layer(self):
        def _lambda(layer):
            return tf.split(layer, self.seq_size, axis=1)

        return _lambda

    def infer_vector(self):
        """Infer a documents vector by training the model against unseen labels
        and text

        Args:
            document (Document): Document to vectorize
        Returns:
            np.array: vector representation of provided document
        """
        raise NotImplementedError("This functionality is not currently implemented.")

    # TODO: Implement check for early stopping!
    def fit(self, epochs):
        """This function trains Keras2Vec with the provided documents

        Args:
            epochs(int): How many times to iterate over the training dataset
        """
        # TODO: Fix weird generator syntax
        self.train_model.fit_generator(self.generator.generator(), steps_per_epoch=1,
                                       epochs=epochs)

        self.doc_embeddings = self.train_model.get_layer('doc_embedding').get_weights()[0]
        self.word_embeddings = self.train_model.get_layer('word_embedding').get_weights()[0]

    def get_doc_embeddings(self):
        """Get the document vectors/embeddings from the trained model
        Returns:
            np.array: Array of document embeddings indexed by encoded doc
        """
        return self.doc_embeddings

    def get_doc_embedding(self, doc):
        """Get the vector/embedding for the provided doc
        Args:
            doc (object): Object used in the inital generation of the model
        Returns:
            np.array: embedding for the provided doc
        """
        enc_doc = self.generator.doc_enc.transform(doc)
        return self.doc_embeddings[enc_doc]

    def get_word_embeddings(self):
        """Get the vectors/embeddings from the trained model
        Returns:
            np.array: Array of embeddings indexed by encoded doc
        """
        return self.word_embeddings

    def get_word_embedding(self, word):
        """Get the vector/embedding for the provided word
        Args:
            word (object): Object used in the inital generation of the model
        Returns:
            np.array: embedding for the provided doc
        """
        enc_word = self.generator.word_enc.transform(word)
        return self.word_embeddings[enc_word]