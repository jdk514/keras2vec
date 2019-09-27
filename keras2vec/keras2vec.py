import tensorflow as tf
import keras.backend as K

from keras.layers import Input, Embedding, Average, Dense, Lambda, Concatenate, Flatten, Dot, Reshape
from keras.optimizers import Adamax, Adam, SGD
from keras.backend import mean
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
        self.model = self.infer_model = None
        self.word_embeddings = self.doc_embeddings = self.infer_embedding = None

        self.generator = DataGenerator(documents, seq_size, neg_sampling)
        # TODO: Fix method for getting model attributes
        self.doc_vocab = len(self.generator.doc_vocab)
        self.label_vocab = len(self.generator.label_vocab)
        self.word_vocab = len(self.generator.text_vocab)
        self.num_labels = len(documents[0].labels)

        self.embedding_size = embedding_size
        self.seq_size = seq_size


    def build_model(self, infer=False):
        """Build both the training and inference models for Doc2Vec"""

        doc_ids = Input(shape=(1,), name='input_docs')
        sequence = Input(shape=(self.seq_size,), name='input_seq')
        labels = Input(shape=(None,), name='input_labels')

        # Setup the embedding layers for training and inference
        doc_inference = Embedding(input_dim=1,
                                  output_dim=self.embedding_size,
                                  input_length=1,
                                  name='inferred_doc')(doc_ids)

        doc_embedding = Embedding(input_dim=self.doc_vocab,
                                  output_dim=self.embedding_size,
                                  input_length=1,
                                  name="doc_embedding")(doc_ids)

        # TODO: Build own embedding layer enabling variable length label input training
        label_embedding = Embedding(input_dim=self.label_vocab,
                                    output_dim=self.embedding_size,
                                    name="label_embedding")(labels)

        seq_embedding = Embedding(input_dim=self.word_vocab,
                                  output_dim=self.embedding_size,
                                  input_length=self.seq_size,
                                  name="word_embedding")(sequence)

        # If the sequence size is greater than 1, we need to average each element
        if self.seq_size > 1:
            split_seq = Lambda(self.__split_layer(self.seq_size), name='split_seq')(seq_embedding)
            avg_seq = Average(name='avg_seq')(split_seq)
        else:
            avg_seq = seq_embedding

        # If the label size is greater than 1, we need to average each element
        avg_labels = Lambda(self.__mean_layer(), name='avg_labels')(label_embedding)
        avg_labels = Reshape((1, 24), name='reshape_labels')(avg_labels)
        #split_label = Lambda(self.__split_layer(label_embedding.shape[1]))(label_embedding)
        #avg_labels = Average()(split_label)

        # If we have labels, we are currently averaging the seq and label data to provide document context
        if self.num_labels > 0:
            context = Average(name='avg_context')([avg_labels, avg_seq])
        else:
            context = avg_seq

        # If we are building the inference model, we'll concat with the inference inputs
        if infer:
            merged = Concatenate(name='merged')([doc_inference, context])
        else:
            merged = Concatenate(name='merged')([doc_embedding, context])

        # We flatten the layer just to ensure that we are dealing with a 1D tensor
        flattened = Flatten(name='flattened')(merged)
        hidden = Dense(self.embedding_size * 3, activation='relu', name='hidden')(flattened)
        output = Dense(1, activation='sigmoid', name='train_output')(hidden)

        # Depending on the nature of the Documents, our model inputs/outputs alter
        if self.num_labels > 0:
            model = Model(inputs=[doc_ids, labels, sequence], outputs=output)
        else:
            model = Model(inputs=[doc_ids, sequence], outputs=output)

        if infer:
            self.infer_model = model
        else:
            self.model = model


    def __mean_layer(self):
        def _lambda(layer):
            return mean(layer, axis=1)

        return _lambda

    def __split_layer(self, splits):
        """Keras Lambda to split a layer, used for the sequence data"""
        def _lambda(layer):
            return tf.split(layer, splits, axis=1)

        return _lambda


    # TODO: Determine if we should remove extra params here, or add them to train_model's fit()
    def infer_vector(self, infer_doc, epochs=5, lr=0.1, init_infer=True):
        """Infer a documents vector by training the model against unseen labels and text.
        Currently inferred vector is passed to an attribute and not returned from this function.

        Args:
            infer_doc (Document): Document for which we will infer a vector
            epochs (int): number of training cycles
            lr (float): the learning rate during inference
            init_infer (bool): determines whether or not we want to reinitalize weights for inference layer
        """
        if self.infer_model is None:
            self.build_model(infer=True)

        tmp_generator = self.generator.get_infer_generator(infer_doc)

        # We want to start with random weights when inferring a new document's vector
        if init_infer:
            sess = K.get_session()
            self.infer_model.get_layer('inferred_doc').embeddings.initializer.run(session=sess)

        # Copy the weights from the trained model into the inference model
        for layer in self.infer_model.layers:
            # Currently using {model_type}_{layer_name} to identify layers to copy
            if layer.name not in ['inferred_doc', 'input_docs']:
                train_weights = self.model.get_layer(layer.name).get_weights()
                layer.set_weights(train_weights)

        # We only want to train the document embedding when inferring
        for layer in self.infer_model.layers:
            layer.trainable = False

        # By only training the inference layer, we ensure that the inference benefits from previous training
        self.infer_model.get_layer('inferred_doc').trainable = True

        optimizer = Adamax(lr=lr)
        self.infer_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        history = self.infer_model.fit_generator(tmp_generator, steps_per_epoch=1, epochs=epochs)

        self.infer_embedding = self.infer_model.get_layer('inferred_doc').get_weights()[0]


    # TODO: Implement check for early stopping!
    def fit(self, epochs, lr=0.1):
        """This function trains Keras2Vec with the provided documents

        Args:
            epochs(int): How many times to iterate over the training dataset
        """
        optimizer = Adamax(lr=lr)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        # TODO: Fix weird generator syntax
        self.model.fit_generator(self.generator.generator(), steps_per_epoch=1,
                                       epochs=epochs)

        self.doc_embeddings = self.model.get_layer('doc_embedding').get_weights()[0]
        self.word_embeddings = self.model.get_layer('word_embedding').get_weights()[0]
        self.label_embeddings = self.model.get_layer('label_embedding').get_weights()[0][1:]


    def get_infer_embedding(self):
        return self.infer_embedding


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


    def get_label_embeddings(self):
        """Get the label vectors/embeddings from the trained model
        Returns:
            np.array: Array of the label embeddings"""
        return self.label_embeddings


    def get_label_embedding(self, label):
        """Get the vector/embedding for the provided label
        Args:
            label (object): Object used in the inital generation of the model
        Returns:
            np.array: embedding for the provided label
        """
        enc_lbl = self.generator.label_enc.transform(label)
        return self.label_embeddings[enc_lbl]


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