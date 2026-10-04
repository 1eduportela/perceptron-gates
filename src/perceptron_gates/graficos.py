import numpy as np
import matplotlib as mp
mp.use("Agg")
import matplotlib.pyplot as plt

def graficar_frontera(neurona,X,y,titulo,ruta):
    """Dibuja la frontera de decisión de la neurona y la guarda en ruta."""
    valores = np.linspace(-0.5,1.5,200)
    xx,yy = np.meshgrid(valores,valores)
    puntos = np.column_stack([xx.ravel(),yy.ravel()])
    probs = neurona.forward(puntos)
    probs = probs.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx,yy,probs,levels=20,cmap="RdYlGn",alpha=0.6)
    ax.contour(xx,yy,probs,levels=[0.5],colors="black",linewidths=2)
    ax.scatter(X[:,0],X[:,1],c=y,cmap="RdYlGn",edgecolors="black",s=150)
    ax.set_title(titulo)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.savefig(ruta)
    plt.close(fig)
