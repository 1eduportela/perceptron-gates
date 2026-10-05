import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

puertas = {
    "OR":   np.array([0, 1, 1, 1], dtype=float),
    "AND":  np.array([0, 0, 0, 1], dtype=float),
    "NAND": np.array([1, 1, 1, 0], dtype=float),
    "NOR":  np.array([1, 0, 0, 0], dtype=float),
    "XOR":  np.array([0, 1, 1, 0], dtype=float),
    "XNOR": np.array([1, 0, 0, 1], dtype=float),
}


def obtener_datos(nombre):
    """Devuelve (X, y) para la compuerta lógica indicada.

    Args:
        nombre: nombre de la compuerta ("OR", "AND", "XOR"...).
                No distingue mayúsculas de minúsculas.

    Returns:
        X: array (4, 2) con las combinaciones de entrada.
        y: array (4,) con la salida esperada para cada fila de X.
    """
    clave = nombre.upper()

    if clave not in puertas:
        raise ValueError(
            f"Compuerta '{nombre}' no válida. Opciones: {list(puertas.keys())}"
        )

    return X.copy(), puertas[clave].copy()

def añadir_producto(X):
    producto = X[:,0]*X[:,1]
    return np.column_stack([X,producto])

if __name__ == "__main__":
    for puerta in puertas:
        X_p, y_p = obtener_datos(puerta)
        print(f"{puerta}: y = {y_p}, X.shape = {X_p.shape}, y.shape = {y_p.shape}")
