import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(1)

class KMeans:
    def __init__(self,k=2):
        self.labels_=None
        self.mu=None
        self.k=k

    def init(self,X,method='kmeans++',random_state=False):
        if method=='kmeans++':
            if random_state is False:
                np.random.seed(0)
            mus=[X[np.random.randint(0,len(X))]]
            while len(mus)<self.k:
                Dxs=[]
                array_mus=np.array(mus)
                for x in X:
                    Dx=np.sum(np.sqrt(np.sum((x-array_mus)**2,axis=1)))
                    Dxs.append(Dx)
                Dxs=np.array(Dxs)
                index=np.argmax(Dxs)
                mus.append(X[index])
            self.mu=np.array(mus)


        elif method=='default':
            self.mu = X[random.sample(range(X.shape[0]), self.k)]

        else:
            raise NotImplementedError

    # p203图9.2算法流程
    def fit(self,X):
        self.init(X,'kmeans++')
        while True:
            C={}
            for i in range(self.k):
                C[i]=[]
            for j in range(X.shape[0]):
                d=np.sqrt(np.sum((X[j]-self.mu)**2,axis=1))
                lambda_j=np.argmin(d)
                C[lambda_j].append(j)
            mu_=np.zeros((self.k,X.shape[1]))
            for i in range(self.k):
                mu_[i]=np.mean(X[C[i]],axis=0)
            if np.sum((mu_-self.mu)**2)<1e-8:
                self.C=C
                break
            else:
                self.mu=mu_
        self.labels_=np.zeros((X.shape[0],),dtype=np.int32)
        for i in range(self.k):
            self.labels_[C[i]]=i

    def predict(self,X):
        preds=[]
        for j in range(X.shape[0]):
            d=np.zeros((self.k,))
            for i in range(self.k):
                d[i]=np.sqrt(np.sum((X[j]-self.mu[i])**2))
            preds.append(np.argmin(d))
        return np.array(preds)

if __name__=='__main__':
    # p202 西瓜数据集4.0
    X=np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
                [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
                [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],
                [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
                [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
                [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]])

    kmeans=KMeans(k=3)
    kmeans.fit(X)
    print(kmeans.C)
    print(kmeans.labels_)
    print(kmeans.predict(X))
    plt.figure(12)
    plt.subplot(121)
    plt.scatter(X[:,0],X[:,1],c=kmeans.labels_)
    plt.scatter(kmeans.mu[:,0],kmeans.mu[:,1],c=range(kmeans.k),marker='+')
    plt.title('tinyml')

    from sklearn.cluster import KMeans
    sklearn_kmeans=KMeans(n_clusters=3)
    sklearn_kmeans.fit(X)
    print(sklearn_kmeans.labels_)
    plt.subplot(122)
    plt.scatter(X[:,0],X[:,1],c=sklearn_kmeans.labels_)
    plt.title('sklearn')
    plt.show()