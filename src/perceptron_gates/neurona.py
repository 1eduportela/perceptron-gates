import numpy as np
from perceptron_gates.datos import obtener_datos

def sigmoid(z):
    return 1/(1+np.exp(-z))

def perdida(p, y):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return np.mean(-(y*np.log(p) + (1 - y)*np.log(1 - p)))
    
class Neurona:
    def __init__(self,n_entradas,semilla=42):
        rng=np.random.default_rng(semilla)
        self.w=rng.uniform(-0.5,0.5,size=n_entradas)
        self.b=rng.uniform(-0.5,0.5)

    def forward(self,X):
        z=X@self.w+self.b
        return sigmoid(z)

    def predecir(self,X,umbral=0.5):
        p=self.forward(X)
        return (p>=umbral).astype(int)

    def gradientes(self,X,y):
        p=self.forward(X)
        er=p-y
        dw=X.T @ er / len(y)
        dm=np.mean(er)
        return dw,dm



if __name__ == "__main__":
    neurona=Neurona(2)
    X,y=obtener_datos("or")
    probabilidades=neurona.forward(X)
    predicciones=neurona.predecir(X)
    print(f"forward:  {np.round(probabilidades, 4)}")
    print(f"predecir: {predicciones}")
    print(f"y:        {y}")
    print(f"pérdida:  {perdida(probabilidades, y):.4f}")
    dw, db = neurona.gradientes(X, y)
    print(f"dw:       {np.round(dw, 4)}")
    print(f"db:       {db:.4f}")
