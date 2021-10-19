class Graph(object):
    def __init__(self, id, neighbours = [], ageNeighbours = []):
        self.id = id
        self.neighbours = neighbours
        self.ageNeighbours = ageNeighbours

    def addNeighbor(self, neighbour, ageNeighbour):
        self.neighbours.append(neighbour)
        self.ageNeighbours.append(ageNeighbour)

    def removeNeighbour(self, neighbour):
        self.ageNeighbours.pop(self.neighbours.index(neighbour))
        self.neighbours.remove(neighbour)

    @property
    def id(self):
        return self.id

    @property
    def neighbors(self):
        return self.neighbours

    @property
    def ageNeighbours(self):
        return self.ageNeighbours