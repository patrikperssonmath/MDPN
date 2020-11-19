
class DataExample:

    def __init__(self, id_name, image, depth, mask, K):
        self.id = id_name
        self.image = image
        self.depth = depth
        self.mask = mask
        self.K = K

    def getId(self):
        return self.id

    def getData():
        return self.id, self.image, self.depth, self.mask, self.K
