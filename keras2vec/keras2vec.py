from keras.layers import Input, Embedding, Average, Dot, Dense
from keras.models import Model

from keras2vec import keras2vec_generator

class Keras2Vec():

    def __init__(self, documents):
        """Load the vector generator with a list of documents

        Args:
            documents (:obj:`list` of :obj:`Document`): List of documents to vectorize
        """

        self.generator = keras2vec_generator(documents)


    def build_model(self):
        """Build the keras model to embed documents"""

        doc_ids = Input(shape=(1,))
        labels = Input(shape=(self.num_labels,))
        sequence = Input(shape=(self.seq_size,))

        doc_inference = Embedding(input=(1, ),
                                  outpu_dim=(self.embedding_size, ),
                                  input_length=1,
                                  name='inferred_vector')(doc_ids)

        doc_embedding = Embedding(input_dim=(self.doc_vocab, ),
                                  outpu_dim=(self.embedding_size, ),
                                  input_length=1,
                                  name="doc_embedding")(doc_ids)

        label_embedding = Embedding(input_dim=(self.label_vocab, ),
                                    outpu_dim=(self.embedding_size, ),
                                    input_length=self.num_labels,
                                    name="label_embedding")(labels)

        seq_embedding = Embedding(input_dim=(self.word_vocab, ),
                                  outpu_dim=(self.embedding_size, ),
                                  input_length=self.seq_size,
                                  name="word_embedding")(sequence)

        context = Average()(label_embedding, seq_embedding)

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

    def fit(self):
        pass

    def fit_generator(self):
        pass
