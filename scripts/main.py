import numpy as np
from perceptron_gates.datos import obtener_datos, puertas
from perceptron_gates.neurona import Neurona

def main():
    pregunta=input("¿Que puerta quieres probar?")
    X,y=obtener_datos(pregunta)
    neurona=Neurona(X.shape[1])
    historial = neurona.entrenar(X, y)
    print(f"forward:  {np.round(neurona.forward(X), 4)}")
    print(f"predecir: {neurona.predecir(X)}")

if __name__ == "__main__":
    main()
