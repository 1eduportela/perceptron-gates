import numpy as np

class Neurona:
    def __init__(self,n_entradas,semilla=42):
        rng=np.random.default_rng(semilla)
        self.w=rng.uniform(-0.5,0.5,size=n_entradas)
        self.b=rng.uniform(-0.5,0.5)


if __name__ == "__main__":
    neurona=Neurona(2)
    print(f"w = {neurona.w}, w.shape = {neurona.w.shape}, b = {neurona.b}")
