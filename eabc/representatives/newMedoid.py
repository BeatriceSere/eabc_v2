# -*- coding: utf-8 -*-

from eabc.representatives import Representative
import scipy.spatial.distance as scpDist
import numpy as np


class newMedoid(Representative):
    
    def __init__(self,initRepr = None,approximateUpdate = True, PoolSize = 20):

        super(newMedoid,self).__init__(initReprElem=initRepr)   
        
        if(not approximateUpdate and not PoolSize):
            raise ValueError("Approximate Update requires PoolSize")

        self._PoolSize = PoolSize
        self._isApprox = approximateUpdate
        
        #Debug
        self._cluster = []
        self._ids = None
        self._minSodIdx = None
            
    def evaluate(self,cluster,Dissimilarity):
        
        #Already have matrix for the cluster                    
        if self._DistanceMatrix is not None:
            M = self._DistanceMatrix
        #First time I have seen the cluster, matrix is None
        else:
            M = Dissimilarity.pdist(cluster)
            
        _ids = np.asarray(range(len(cluster)))

        #check M is in a condensed form. We will use a square Matrix
        if scpDist.is_valid_y(M):
            M = scpDist.squareform(M)            
        #debug
        assert(M.shape[0] == M.shape[1])            

        if self._isApprox:

            #We want to fix cluster size to pool size            
            if len(cluster) <= self._PoolSize:
                # Evaluate all distances even if we have a diss matrix?
                M = Dissimilarity.pdist(cluster)
                if scpDist.is_valid_y(M):
                    M = scpDist.squareform(M)            
                #debug
                assert(M.shape[0] == M.shape[1])
                self._cluster = cluster                              
                #Testing - We can use cluster size differences assuming last patterns are new
                #newP = np.setdiff1d(_ids,self._ids)
                #TODO: Assuming non-metric
                # v_h = [Dissimilarity(cluster[newP],self._cluster[u]) for u in self._ids]
                # v_v = [Dissimilarity(self._cluster[u],cluster[newP]) for u in self._ids]
                # v = 0.5*(np.asarray(v_h) + np.asarray(v_v))
                    
                # M = np.vstack((M,v))
                # v.append(0)
                # M = np.hstack((M,v))
                 
                #Retain all data               
            
            else:
                #randomly choose two pattern from the pool
                id_p1, id_p2 = np.random.choice(self._PoolSize,2,replace=False)
#                id_p1, id_p2 = np.random.choice(len(self._cluster),2,replace=False)
                d1 = self._DistanceMatrix[self._minSodIdx,id_p1]
                d2 = self._DistanceMatrix[self._minSodIdx,id_p2]                             
                #Farthest pattern from medoid will be changed 
                id_p = id_p1 if d1>=d2 else id_p2
                
                #set of cluster ids without the pattern id to remove
                #diffIds = np.setdiff1d(self._ids,id_p)
                diffIds = np.setdiff1d(range(self._PoolSize),id_p)                
                #New Id to be introduced
                newP = np.setdiff1d(_ids,self._ids)[0]
                
                #Change old pattern in the container with the newest 
                self._cluster[id_p] = cluster[newP]
                
                v_h = np.zeros((self._PoolSize,))
                v_v = np.zeros((self._PoolSize,))
                for u in diffIds:
                    v_h[u]=Dissimilarity(cluster[newP], self._cluster[u])
                    v_v[u]=Dissimilarity(self._cluster[u],cluster[newP])
                    
                # v_h = [Dissimilarity(cluster[newP], self._cluster[u]) for u in diffIds]
                # v_v = [Dissimilarity(self._cluster[u],cluster[newP]) for u in diffIds]
                v = 0.5*(np.asarray(v_h) + np.asarray(v_v))
                
                M[id_p,:] = v
                M[:,id_p] = v

                
        SoD = np.sum(M, axis = 0)    

        self._minSodIdx = np.argmin(SoD)        
        self._representativeElem = cluster[self._minSodIdx]    
        self._SOD = SoD[self._minSodIdx]
        self._ids = _ids
        self._DistanceMatrix = M
        self._cluster = cluster
              
        
        ####    
        # pairwise_dist = Dissimilarity.pdist(cluster)
    
        # if scpDist.is_valid_y(pairwise_dist):
        #     pairwise_dist = scpDist.squareform(pairwise_dist)
        
        # self._DistanceMatrix = pairwise_dist
        
        # #FIXME: diss matrix for graphs is not symmetric
        # SoD = np.sum(pairwise_dist, axis = 0)    
        # minSoDIdx = np.argmin(SoD)
        
        # self._representativeElem = cluster[minSoDIdx]    
        # self._SOD = SoD[minSoDIdx]                