from numpy import array, linalg
from numpy import dot
from random import random
from yaplf.algorithms import IterativeAlgorithm

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
    #ricordiamoci di scrivere che lancia ValueError

        i = self.points.index(point)
        return memberships[i]
    
    def getPoints(self):
        return points

    def getCentroid(self , m =2):
        #return dot(self.points, self.memberships)/sum(self.memberships)
        centroid = [0] * len(self.points[0].pattern)
        den = 0.0
        data = array([elem.pattern for elem in self.points])
        for (d, membership) in array((data, self.memberships)).T:
            centroid += d * membership ** m
            den += membership**m

        return centroid/den

class Clusterer(object):
    def __init__(self, **kwargs):
        self.clusters = None

    def compute(self, example):
        pass
    def clusters(self):
        pass
    def getCentroids(self):
        pass

class FuzzyCMeansClusterer(Clusterer):
    def __init__(self, points, memberships):
        self.points = points
        self.memberships = memberships

    def clusters(self):
        return [FuzzyCMeansCluster(self.points, m) for m in array(self.memberships).transpose()]

    def compute(self, point, **kwargs):
    #ricordiamoci di scrivere che lancia ValueError

        try:
            method = kwargs['method']
        except KeyError:
            method = 'fuzzy'

        if method == 'fuzzy':
            return [(c, c.compute(point)) for c in self.clusters()]
        elif method == 'highest':
            results = [(c, c.compute(point)) for c in self.clusters()]
            results.sort(key = lambda x: x[1], reverse=True)
            return results[0]

    def getCentroids(self, m = 2):
        return [elem.getCentroid(m = m) for elem in self.clusters()]


class FuzzyCMeansAlgorithm(IterativeAlgorithm):
    def __init__(self, sample, n_clusters = 2, m = 2,  **kwargs ):
        
        IterativeAlgorithm.__init__(self, sample)

        try:
            self.initializer = kwargs['initializer']
        except KeyError:
            self.initializer = False
        
        self.n_clusters = n_clusters
        self.m = m
        self.distance = 1.0
        #self.reset(**kwargs)

        self.memberships_old = None
        self.reset(**kwargs)

    def matrix_copy(self , matrix):
        copy = []
        for row in matrix:
            copy.append(row[:])
        return copy
    
    def normalize(self, vector):
        normalization_coeff = sum(vector)
        return [e/normalization_coeff for e in vector]

    def get_current_iteration_value(self):
        return self.model.memberships

    def get_previous_iteration_value(self):
        return self.memberships_old   

    def run(self, **kwargs):
        IterativeAlgorithm.run(self, **kwargs)
        data = array([elem.pattern for elem in self.sample])
        while self.stop_criterion.stop() == False:
            self.memberships_old = self.matrix_copy(self.model.memberships)
            #print self.model.memberships
            centroids = self.model.getCentroids(m= self.m)
            #print centroids
            #exit()

            for j in range(len(centroids)):
                for i in range(len(data)):
                    den = sum([(linalg.norm(data[i] - centroids[j])/
                        linalg.norm(data[i]-centroids[k]))**(2/(self.m-1)) 
                        for k in range(self.n_clusters)])
                    self.model.memberships[i][j] = 1/den
            
        self.notify_observers()


    def reset(self, **kwargs):
        IterativeAlgorithm.reset(self)
        # memberships = [self.normalize([random() for j in range(self.n_clusters)]) for i in range(len(self.sample))]
        # self.model = FuzzyCMeansClusterer(self.sample, memberships) 

        if self.initializer:
            self.n_clusters = len(self.initializer)
            centroids = self.initializer
            data = array([elem.pattern for elem in self.sample])
            memberships = [self.normalize([random() for j in range(self.n_clusters)]) for i in range(len(self.sample))]
            for j in range(len(centroids)):
                for i in range(len(data)):
                    den = sum([(linalg.norm(data[i] - centroids[j])/
                        linalg.norm(data[i]-centroids[k]))**(2/(self.m-1)) 
                        for k in range(self.n_clusters)])
                    memberships[i][j] = 1/den
        else:
            memberships = [self.normalize([random() for j in range(self.n_clusters)]) for i in range(len(self.sample))]
        
        self.model = FuzzyCMeansClusterer(self.sample, memberships) 
            
            
        # 10 means simply a big value, so that we don't run the risk of erronously stop
        # if we check convergence before first iteration
        
        self.memberships_old = [[10 for j in range(self.n_clusters)] for i in range(len(self.sample))]


