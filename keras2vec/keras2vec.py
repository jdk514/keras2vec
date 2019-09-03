from keras.layers import Input, Embedding, Average, Dot, Dense
from keras.models import Model

from keras2vec.data_generator import DataGenerator


# TODO: Fix naming convention between words, text, labels, docs
class Keras2Vec:

    def __init__(self, documents, embedding_size=16, seq_size=3):
        """Load the vector generator with a list of documents

        Args:
            documents (:obj:`list` of :obj:`Document`): List of documents to vectorize
        """

        self.generator = DataGenerator(documents)
        # TODO: Fix method for getting model attributes
        self.doc_vocab = len(self.generator.doc_vocab)
        self.label_vocab = len(self.generator.label_vocab)
        self.word_vocab = len(self.generator.text_vocab)
        self.num_labels = len(documents[0].labels)

        self.embedding_size = embedding_size
        self.seq_size = seq_size


    def build_model(self):
        """Build the keras model to embed documents"""

        doc_ids = Input(shape=(1,))
        labels = Input(shape=(self.num_labels,))
        sequence = Input(shape=(self.seq_size,))

        doc_inference = Embedding(input_dim=1,
                                  output_dim=self.embedding_size,
                                  input_length=1,
                                  name='inferred_vector')(doc_ids)

        doc_embedding = Embedding(input_dim=self.doc_vocab,
                                  output_dim=self.embedding_size,
                                  input_length=1,
                                  name="doc_embedding")(doc_ids)

        label_embedding = Embedding(input_dim=self.label_vocab,
                                    output_dim=self.embedding_size,
                                    input_length=self.num_labels,
                                    name="label_embedding")(labels)

        seq_embedding = Embedding(input_dim=self.word_vocab,
                                  output_dim=self.embedding_size,
                                  input_length=self.seq_size,
                                  name="word_embedding")(sequence)

        context = Average()([label_embedding, seq_embedding])

        # Build training model
        train_merged = Dot()(doc_embedding, context)
        train_output = Dense(1, activation='sigmoid')(train_merged)
        train_model = Model(inputs=[doc_embedding, label_embedding, seq_embedding],
                            outputs=train_output)

        train_model.compile(optimizer='adamax',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
        self.train_model = train_model


        # Build model for inference
        infer_merged = Dot()(doc_inference, context)
        infer_output = Dense(1, activation='sigmoid')(infer_merged)
        infer_model = Model(inputs=[doc_inference, label_embedding, seq_embedding],
                            outputs=infer_output)

        infer_model.compile(optimizer='adamax',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
        self.train_model = infer_model


    def infer_vector(self):
        """Infer a documents vector by training the model against unseen labels
        and text

        Args:
            document (Document): Document to vectorize
        Returns:
            np.array: vector representation of provided document
        """

        pass

    def fit(self, epochs):
        self.train_model.fit_generator(self.generator, steps_per_epcoh=1,
                                       epochs=epochs)

    def fit_generator(self):
        pass
