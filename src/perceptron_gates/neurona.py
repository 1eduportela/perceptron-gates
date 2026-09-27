import numpy as np
from perceptron_gates.datos import obtener_datos

def sigmoid(z):
    return 1/(1+np.exp(-z))

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

if __name__ == "__main__":
    neurona=Neurona(2)
    X,y=obtener_datos("or")
    probabilidades=neurona.forward(X)
    predicciones=neurona.predecir(X)
    print(f"forward:  {np.round(probabilidades, 4)}")
    print(f"predecir: {predicciones}")
    print(f"y:        {y}")
