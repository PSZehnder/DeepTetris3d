import numpy as np

# Array representations of our pieces

st = np.array([[[1, 1, 1, 1]]])

l = np.array([[[1, 1, 1],
              [1, 0, 0]]])

s = np.array([[[1, 1, 0],
             [0, 1, 1]]])

# shape(3,
sq = np.array([[[1, 1],
                [1, 1]]])

# shape (3, 2, 1)
t = np.array([[[1, 1, 1],
               [0, 1, 0]]])

# shape: (2, 2, 2)
rs = np.array([[[1, 1],
               [0, 1]],
              [[0, 0],
              [0, 1]]])

# shape: (2, 2, 2)
ls = np.array([[[1, 1],
               [1, 0]],
              [[0, 0],
              [1, 0]]])

# shape: (2, 2, 2)
b = np.array([[[1, 1],
               [0, 1]],
              [[0, 1],
              [0, 0]]])


# standardize the matricies by embedding them in a matrix of embedding_shape
def embed(matrix, embedding_shape=(2, 2, 4)):
    embedding_matrix = np.zeros(shape=embedding_shape)
    embedding_matrix[0:matrix.shape[0], 0:matrix.shape[1], 0:matrix.shape[2]] = matrix
    return embedding_matrix


# generate a random block that fits within the bounding shape with density proportion of them filled (average)
# note that by construction: embed(generate_random_block(bounding_shape)) == generate_random_block(bounding_shape)
def generate_random_block(bounding_shape, density=0.25):
    matrix = np.zeros(bounding_shape)
    current = [0, 0, 0]
    done = False
    while not done:
        matrix[tuple(current)] = 1
        axis = np.random.choice(range(4), p=[(1 - density)/3] * 3 + [density])
        if axis == 3 or not (matrix == 0).any():
            return matrix
        direction = np.random.choice([-1, 1])
        if current[axis] + direction in range(bounding_shape[axis]):
            current[axis] += direction




class Tetromino:

    def __init__(self, matrix, location, embedding=(2, 2, 4)):
        self.location = np.array(location)
        self.matrix = embed(matrix, embedding_shape=embedding)
        self.color = np.random.randint(1, len(color_vec) - 1)
        self.shape = matrix.shape
        self.total_dimension = 1
        for d in embedding:
            self.total_dimension *= d


shapelib = {
    'Straight': st,
    'L': l,
    'S': s,
    'Square': sq,
    'T': t,
    'RScrew': rs,
    'LScrew': ls,
    'Branch': b
}

color_vec = ("r", "g", "b", "c", "m", "k", (1, 0.5, 0.5), (0.5, 0.5, 1), (0.75, 0.25, 0.1))
