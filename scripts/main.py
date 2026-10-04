import numpy as np
from perceptron_gates.datos import obtener_datos, puertas
from perceptron_gates.neurona import Neurona
from perceptron_gates.graficos import graficar_frontera
from datetime import datetime

def main():
    pregunta=input("¿Que puerta quieres probar?")
    X,y=obtener_datos(pregunta)
    neurona=Neurona(X.shape[1])
    historial = neurona.entrenar(X, y)
    print(f"forward:  {np.round(neurona.forward(X), 4)}")
    print(f"predecir: {neurona.predecir(X)}")

    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta = f"figuras/frontera_{pregunta.lower()}_{fecha}.png"
    graficar_frontera(neurona, X, y, f"Puerta: {pregunta.upper()}", ruta)
    print(f"Gráfica guardada en {ruta}")

if __name__ == "__main__":
    main()
