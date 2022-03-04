import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from scipy.stats import zscore
import networkx as nx


class ChaoticNeuralNetwork:

    def __init__(self, N, tau, g):
        self.N = N  # N Neurons
        self.tau = tau  # Time constant
        self.g = g  # Chaos parameter
        self.A = np.zeros((N, N))  # Adjacency matrix
        r_0 = np.random.uniform(-1, 1, N)  # Initial firing rates
        self.R = np.expand_dims(r_0, axis=1)  # Firing rates (time series)

    @property
    def r(self):
        return np.expand_dims(self.R[:, -1], axis=1)

    def wireNeurons(self, adjacencyMatrix):
        self.A = adjacencyMatrix

    def simulate(self, duration, step=1):
        T = int(duration / step) + 1
        t = np.linspace(0, duration, T)
        R = np.zeros((self.N, T))
        R[:, 0] = self.R[:, -1]
        for i in range(1, T):
            R[:, i] = R[:, i - 1] + step * self.dr(R[:, i - 1])
        self.R = np.append(self.R, R[:, 1:], axis=1)

    def dr(self, r):
        dr = (1 / self.tau) * (-r + (self.g * np.dot(self.A, np.tanh(r))))
        return dr

    def displayTimeSeries(self, Nneurons, linewidth=3, figsize=(15, 10)):
        plt.figure(figsize=figsize)
        ncol = 2
        IDs = (self.N * np.random.uniform(0, 1, Nneurons)).astype('int')
        for i in range(Nneurons):
            plt.plot(zscore(self.R[IDs[i], :]) + (5 * i), color='black', linewidth=linewidth)
        plt.ylim([-5, (Nneurons + 1) * 5])
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

    def displayRaster(self, figsize=(8, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        image = plt.imshow(self.R, cmap='hot', extent=[0, 1.5, 1, 0], interpolation=None)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(image, cax=cax)
        cbar.ax.set_ylabel('Firing rate')
        ax.set_xlabel('Time')
        ax.set_ylabel('Neurons')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(pad=0)
        plt.show()


def randomNetwork(N, rho):
    G = nx.erdos_renyi_graph(N, rho)
    weightedMatrix = nx.to_numpy_array(G)
    IDs = np.where(weightedMatrix == 1)
    for i in range(len(IDs[0])):
        weightedMatrix[IDs[0][i], IDs[1][i]] = np.random.normal(0, (1 / ((N * rho) ** (1/3))))
    return weightedMatrix


def modularNetwork(moduleSizes, p_intra, p_inter):
    Nmodules = len(moduleSizes)
    probabilityMatrix = np.zeros((Nmodules, Nmodules))
    for i in range(probabilityMatrix.shape[0]):
        for j in range(probabilityMatrix.shape[0]):
            if i == j:
                probabilityMatrix[i, j] = p_intra[i]
            else:
                probabilityMatrix[i, j] = p_inter
    G = nx.stochastic_block_model(moduleSizes, probabilityMatrix)
    weightedMatrix = nx.to_numpy_array(G)
    IDs = np.where(weightedMatrix == 1)
    rho = np.mean(p_intra)
    for i in range(len(IDs[0])):
        weightedMatrix[IDs[0][i], IDs[1][i]] = np.random.normal(0, (1 / ((N * rho) ** (1/3))))
    return weightedMatrix


def BANetwork(N, M):
    G = nx.barabasi_albert_graph(N, M)
    weightedMatrix = nx.to_numpy_array(G)
    IDs = np.where(weightedMatrix == 1)
    rho = len(IDs[0]) / ((N ** 2) - N)
    for i in range(len(IDs[0])):
        weightedMatrix[IDs[0][i], IDs[1][i]] = np.random.normal(0, (1 / ((N * rho) ** (1/3))))
    return weightedMatrix


def WSNetwork(N, k, p):
    G = nx.watts_strogatz_graph(N, k, p)
    weightedMatrix = nx.to_numpy_array(G)
    IDs = np.where(weightedMatrix == 1)
    rho = len(IDs[0]) / ((N ** 2) - N)
    for i in range(len(IDs[0])):
        weightedMatrix[IDs[0][i], IDs[1][i]] = np.random.normal(0, (1 / ((N * rho) ** (1/3))))
    return weightedMatrix


def circulantNetwork(N, offsets):
    G = nx.circulant_graph(N, offsets)
    weightedMatrix = nx.to_numpy_array(G)
    IDs = np.where(weightedMatrix == 1)
    rho = len(IDs[0]) / ((N ** 2) - N)
    for i in range(len(IDs[0])):
        weightedMatrix[IDs[0][i], IDs[1][i]] = np.random.normal(0, (1 / ((N * rho) ** (1/3))))
    return weightedMatrix


def animateNetwork(network, Nframes, step, nodeSize=50):

    G = nx.from_numpy_array(np.abs(network.A))
    pos = nx.spring_layout(G)  # positions for all nodes
    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=nodeSize, node_color=network.R[:, 0], cmap='hot')
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.1)
    x, y = ax.get_xlim()[0], ax.get_ylim()[1]
    title = plt.text(x + 0.15 * np.abs(x), y - 0.1 * np.abs(y), r"$t = {:.0f}$".format(0), ha="center")
    plt.axis('off')
    plt.tight_layout(pad=0)

    def drawframe(i):
        nodes = nx.draw_networkx_nodes(G, pos, node_size=nodeSize, node_color=network.R[:, i * step], cmap='hot')
        title.set_text(r"$t = {:.0f}$".format(i * step))
        return nodes, title,

    anim = animation.FuncAnimation(fig, drawframe, frames=Nframes, interval=10, blit=True)
    plt.show()


if __name__ == '__main__':

    N = 500  # Neurons
    tau = 10  # Time constant (ms)
    g = 1.5  # Chaos

    #adjacencyMatrix = modularNetwork([250] + [50] * 5, [0.75] + [0.75] * 5, 0.01)
    #adjacencyMatrix = randomNetwork(N, 0.1)
    adjacencyMatrix = BANetwork(N, 20)
    #adjacencyMatrix = WSNetwork(N, 10, 0.1)

    network = ChaoticNeuralNetwork(N, tau, g)
    network.wireNeurons(adjacencyMatrix)
    network.simulate(5000)

    network.displayTimeSeries(20)

    animateNetwork(network, 1000, 4, nodeSize=75)
