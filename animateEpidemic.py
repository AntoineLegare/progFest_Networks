import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx
plt.rcParams.update({'font.size': 20})


class Epidemic:

    def __init__(self, N, I_0=1):
        self.N = N
        self.status = np.expand_dims(np.array([0] * N), axis=1)
        self.status[:I_0, 0] = 1
        np.random.shuffle(self.status)
        self.A = np.zeros((N, N))

    def initiateContacts(self, adjacencyMatrix):
        self.A = adjacencyMatrix

    def simulate(self, duration, p_infection=0.1, p_recovery=0.01):
        T = int(duration)
        status = np.zeros((self.N, T)).astype('int')
        status[:, 0] = self.status[:, -1]
        for i in range(1, T):
            infectedPeople = status[:, i - 1] == 1
            status[infectedPeople, i] = 1
            infectedContacts = p_infection * (self.A @ status[:, i - 1])
            immuneResponse = np.random.uniform(0, 1, self.N)
            infections = infectedContacts > immuneResponse
            status[infections, i] = 1
            condition1 = np.random.uniform(0, 1, self.N) > (1 - p_recovery)
            condition2 = (status[:, i - 1] == 1)
            recoveries = condition1 & condition2
            status[recoveries, i] = 0
        self.status = np.append(self.status, status[:, 1:], axis=1)


def WSNetwork(N, k, p):
    G = nx.watts_strogatz_graph(N, k, p)
    matrix = nx.to_numpy_array(G)
    return matrix


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
    matrix = nx.to_numpy_array(G)
    return matrix


def animateEpidemic(epidemic, Nframes, fps=5, nodeSize=50):

    G = nx.from_numpy_array(np.abs(epidemic.A))
    pos = nx.spring_layout(G)  # positions for all nodes
    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=nodeSize, node_color=epidemic.status[:, 0], cmap='Reds',
                                       vmin=0, vmax=1)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.1)
    x, y = ax.get_xlim()[0], ax.get_ylim()[1]
    title = plt.text(x + 0.15 * np.abs(x), y - 0.1 * np.abs(y), r"$t = {:.0f}$".format(0), ha="center")
    plt.axis('off')
    plt.tight_layout(pad=0)

    def drawframe(i):
        nodes = nx.draw_networkx_nodes(G, pos, node_size=nodeSize, node_color=epidemic.status[:, i], cmap='Reds',
                                       vmin=0, vmax=1)
        title.set_text(r"$t = {:.0f}$".format(i))
        return nodes, title,

    anim = animation.FuncAnimation(fig, drawframe, frames=Nframes, interval=int(1000/fps), blit=True)
    plt.show()


if __name__ == '__main__':

    N = 500  # People
    I_0 = 50 # Initial number of infected people
    p_infection = 0.017
    p_recovery = 0.1
    duration = 1000
    fps = 30

    adjacencyMatrix = WSNetwork(N, 10, 0.2)
    # adjacencyMatrix = modularNetwork([int(N/2)] + [int(N/10)] * 5, [0.05] + [0.1] * 5, 0.005)

    epidemic = Epidemic(N, I_0=I_0)
    epidemic.initiateContacts(adjacencyMatrix)
    epidemic.simulate(duration, p_infection=p_infection, p_recovery=p_recovery)

    animateEpidemic(epidemic, duration, fps, nodeSize=50)

    plt.figure(figsize=(7, 5))
    plt.plot(np.mean(epidemic.status, axis=0), linewidth=2, color='black')
    plt.ylabel('Fraction of infected people')
    plt.xlabel('Time steps')
    plt.tight_layout(pad=0)
    plt.ylim([0, 1])
    plt.xlim([0, duration])
    plt.show()