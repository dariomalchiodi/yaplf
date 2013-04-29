
from numpy import dot

class Cluster( object ):
    def __init__(self, **kwargs):
        pass

    def compute(self, point):
        pass
        
    def getCentroid(self):       
        pass

    def getPoints(self):
        pass

class FuzzyCMeansCluster(Cluster):
    def __init__(self, points, memberships, **kwargs):
        Cluster.__init__(self, **kwargs)
        self.points = points
        self.memberships = memberships

    def compute(self, point):
    '''ricordiamoci di scrivere che lancia ValueError
    '''

        i = self.points.index(point)
        return memberships[i]

    def getPoints(self):
        return points

    def getCentroid(self):
        return dot(self.points, self.memberships)/sum(self.memberships)

class Clusterer(object):
    def __init__(self, **kwargs):
        self.clusters = None

    def compute(self, example):
        pass


class FuzzyCMeansClusterer(Clusterer):
    def __init__(self, points, memberships):
        self.points = points
        self.memberships = memberhips

    def clusters(self):
        return [FuzzyCMeansCluster(self.points, m) for m in self.memberships]

    def compute(self, point, **kwargs):
    '''ricordiamoci di scrivere che lancia ValueError
    '''

        try:
            method= kwargs['method']
        except KeyError:
            method = 'fuzzy'

        if method == 'fuzzy':
            return [(c, c.compute(point)) for c in self.clusters()]
        elif method == 'highest':
            results = [(c, c.compute(point)) for c in self.clusters()]
            results.sort(key = lambda x: x[1], reverse=True)
            return results[0]





