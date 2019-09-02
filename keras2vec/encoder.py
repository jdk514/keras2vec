
class Encoder():
    def __init__(self, items):
        """Take in items to encode
        Args:
            items (:obj:`list` of objects)"""
        self.encoder = self.inverse_encoder = None
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
            index (int): index to reverse encoding
        Returns:
            object: decoded object"""

        try:
            item = self.inverse_encoder[index]
        except KeyError:
            raise KeyError("Provided encoding does not exist.")

        return item
