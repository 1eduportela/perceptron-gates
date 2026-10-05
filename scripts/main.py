import numpy as np
from perceptron_gates.datos import obtener_datos, puertas, añadir_producto
from perceptron_gates.neurona import Neurona
from perceptron_gates.graficos import graficar_frontera
from datetime import datetime

def main():
    pregunta=input("¿Que puerta quieres probar?")
    X, y = obtener_datos(pregunta)

    while True:
        entradas = input("¿Cuántas entradas quieres en la neurona? (2 o 3): ").strip()
        if entradas not in ("2", "3"):
            print("Número no válido, escribe 2 o 3.")
            continue
        break

    usar_producto = entradas == "3"
    if usar_producto:
        X = añadir_producto(X)

    neurona=Neurona(X.shape[1])
    historial = neurona.entrenar(X, y)
    print(f"forward:  {np.round(neurona.forward(X), 4)}")
    print(f"predecir: {neurona.predecir(X)}")

    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta = f"figuras/frontera_{pregunta.lower()}_{pregunta2.lower()}ins_{fecha}.png"
    graficar_frontera(neurona, X, y, f"Puerta: {pregunta.upper()}", ruta)
    print(f"Gráfica guardada en {ruta}")

if __name__ == "__main__":
    main()
