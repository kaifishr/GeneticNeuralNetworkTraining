import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib import cm
from scipy.special import expit
from scipy.interpolate import griddata
from util.data import DataLoader

np.random.seed(42)

class NeuralNetwork(object):

    def __init__(self, problem_type, n_epochs, problem, n_points, noise_level, network_layers, n_outputs, n_agents,
                 mutation_rate, mutation_prob, mutation_type, crossover_type, batch_size, activation_fct, save_path):

        self.problem_type = problem_type
        self.n_epochs = n_epochs
        self.network_layers = network_layers
        self.n_layers = len(self.network_layers)
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.activation_fct = activation_fct

        # Training parameter
        self.mutation_rate = mutation_rate
        self.mutation_prob = mutation_prob
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type

        # Generate data
        data_loader = DataLoader(problem=problem, n_outputs=n_outputs)
        self.x_train, self.y_train, self.x_test, self.y_test = \
            data_loader.get_data(n_points=n_points, noise_level=noise_level)
        self.x_train_plt, self.y_train_plt, self.x_test_plt, self.y_test_plt = \
            data_loader.get_data(n_points=2000, noise_level=noise_level)

        # Initialize weights
        self.W = [[self.kaiming(l, self.network_layers) for l in range(self.n_layers-1)] for p in range(self.n_agents)]
        self.B = [[np.zeros((network_layers[l+1])) for l in range(self.n_layers-1)] for p in range(self.n_agents)]

        # Activation function
        self.act_function = self.activation_function()

        # Mutation counter
        self.n = self.n_layers - 2

        # Other parameters
        self.best_agent = 0
        self.cost = None

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.print_info()

        if self.problem_type == 'classification':
            self.plot_dataset()

    def print_info(self):
        fp = open(self.save_path + 'params.log', 'w')
        prototype = 'problem_type={}\nn_epochs={}\nnetwork_layers={}\nn_agents={}\nbatch_size={}\nactivation_fct={} \
                     \nmutation_rate={}\nmutation_prob={}\nmutation_type={}\ncrossover_type={}'
        fp.write(prototype.format(self.problem_type,
                                  self.n_epochs,
                                  self.network_layers,
                                  self.n_agents,
                                  self.batch_size,
                                  self.activation_fct,
                                  self.mutation_rate,
                                  self.mutation_prob,
                                  self.mutation_type,
                                  self.crossover_type))
        fp.close()
        print(prototype.format(self.problem_type,
                               self.n_epochs,
                               self.network_layers,
                               self.n_agents,
                               self.batch_size,
                               self.activation_fct,
                               self.mutation_rate,
                               self.mutation_prob,
                               self.mutation_type,
                               self.crossover_type))

    def discretize(self, x, eta=10.0):
        return np.round(eta * x) / eta

    def dsin(self, x):
        return self.discretize(np.sin(x))

    def dtanh(self, x):
        return self.discretize(np.tanh(x))

    def drelu(self, x):
        return self.discretize(self.relu(x))

    @staticmethod
    def relu(x):
        return x * (x > 0.0)

    @staticmethod
    def heaviside(x):
        return 1.0 * (x > 0.0)

    @staticmethod
    def sign(x):
        return np.where(x > 0.0, 1.0, -1.0)

    @staticmethod
    def floor(x, d=1.0):
        # Goes to x for d -> inf
        x = np.floor(d * x) / d
        return x

    @staticmethod
    def sawtooth(x, d=1.0):
        return d*x - np.floor(d*x)

    def activation_function(self):
        if self.activation_fct == 'relu':
            return self.relu
        elif self.activation_fct == 'heaviside':
            return self.heaviside
        elif self.activation_fct == 'sign':
            return self.sign
        elif self.activation_fct == 'floor':
            return self.floor
        elif self.activation_fct == 'sin':
            return np.sin
        elif self.activation_fct == 'tanh':
            return np.tanh
        elif self.activation_fct == 'sigmoid':
            return expit
        elif self.activation_fct == 'sawtooth':
            return self.sawtooth
        elif self.activation_fct == 'dtanh':
            return self.dtanh
        elif self.activation_fct == 'dsin':
            return self.dsin
        elif self.activation_fct == 'drelu':
            return self.drelu
        else:
            raise Exception('Activation function not implemented.')

    @staticmethod
    def kaiming(l, network_layers):
        return np.random.normal(size=(network_layers[l], network_layers[l+1])) * np.sqrt(2.0 / network_layers[l])

    def grouped_rand_idx(self):
        idx = np.random.permutation(len(self.x_train))
        return [idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size)]

    def prediction(self, p, x_data):
        x = np.copy(x_data)
        for l in range(self.n_layers - 2):
            x = np.dot(x, self.W[p][l]) + self.B[p][l]
            x = self.act_function(x)
        y_pred = np.dot(x, self.W[p][-1]) + self.B[p][-1]
        return y_pred

    def compute_cost(self, y_batch, y_pred):
        return np.sum((y_batch - y_pred) ** 2) / self.batch_size

    def feedforward(self, idx):
        cost = []
        x_batch, y_batch = self.x_train[idx], self.y_train[idx]
        for p in range(self.n_agents):
            y_pred = self.prediction(p, x_batch)
            cost.append(self.compute_cost(y_batch, y_pred))
        self.cost = cost
        self.best_agent = np.argmin(self.cost)

    def comp_accuracy(self):
        y_pred = self.prediction(self.best_agent, self.x_test)
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(y_pred)
        return accuracy

    def logger(self, epoch, fp):
        if self.problem_type == 'regression':
            fp.write('{} {} {} {}\n'.format(epoch, self.cost[self.best_agent], self.mutation_rate, self.mutation_prob))
            print('{} {} {} {}'.format(epoch, self.cost[self.best_agent], self.mutation_rate, self.mutation_prob), flush=True)
        elif self.problem_type == 'classification':
            accuracy = self.comp_accuracy()
            fp.write('{} {} {} {} {}\n'.format(epoch, self.cost[self.best_agent], accuracy, self.mutation_rate, self.mutation_prob))
            print('{} {} {} {} {}'.format(epoch, self.cost[self.best_agent], accuracy, self.mutation_rate, self.mutation_prob), flush=True)
        else:
            raise Exception('Problem type not implemented')
        fp.flush()

    def visualizer(self, epoch):
        if self.problem_type == 'regression':
            self.plot_progress_regression(epoch)
        elif self.problem_type == 'classification':
            self.plot_progress_classification(epoch, vis_type='prediction')
            self.plot_progress_classification(epoch, vis_type='classification')
        else:
            raise Exception('Problem type not implemented')

    def run(self):
        # Open log file
        fp = open(self.save_path + 'data.log', 'w')

        # Initial prediction
        idx = list(range(len(self.x_train)))
        self.feedforward(idx)

        # Initial logging
        epoch = 0
        self.logger(epoch, fp)
        self.visualizer(epoch)

        cost_old = self.cost[self.best_agent]

        for epoch in range(1, self.n_epochs+1):

            # Optimization step
            idx_list = self.grouped_rand_idx()
            for idx in idx_list:
                self.feedforward(idx)
                self.crossover()
                self.mutation()

            # Print progress to file every epoch
            if epoch % 50 == 0:
                self.logger(epoch, fp)

            # Visualize progress
            if epoch % 1000 == 0:
                self.visualizer(epoch)

        # Close log file
        fp.close()

    def mutation(self):

        if self.mutation_type == 'default':
            self.W = [[self.W[p][l] + np.random.uniform(-self.mutation_rate, self.mutation_rate, size=self.W[p][l].shape) * \
                       (np.random.random(self.W[p][l].shape) < self.mutation_prob) for l in range(self.n_layers - 1)]
                      for p in range(self.n_agents)]
            self.B = [[self.B[p][l] + np.random.uniform(-self.mutation_rate, self.mutation_rate, size=self.B[p][l].shape) * \
                       (np.random.random(self.B[p][l].shape) < self.mutation_prob) for l in range(self.n_layers - 1)]
                      for p in range(self.n_agents)]

        elif self.mutation_type == 'default_v2':
            # Efficient mutation operation for large networks
            for l in range(self.n_layers - 1):
                rows, cols = self.W[0][l].shape
                sample_size = int(self.mutation_prob * self.W[0][l].size)
                if sample_size == 0:
                    sample_size = np.random.binomial(1, self.mutation_prob)
                for p in range(self.n_agents):
                    self.W[p][l][np.random.randint(rows, size=sample_size), np.random.randint(cols, size=sample_size)] += \
                        np.random.uniform(-self.mutation_rate, self.mutation_rate, size=sample_size)
                    self.B[p][l][np.random.randint(cols, size=sample_size)] += np.random.uniform(-self.mutation_rate, self.mutation_rate,
                                                                                                 size=sample_size)

        elif self.mutation_type == 'random_cycle':
            # Change weights of a random layer
            for p in range(self.n_agents):
                l = np.random.randint(self.n_layers - 1)
                self.W[p][l] = self.W[p][l] + np.random.uniform(-self.mutation_rate, self.mutation_rate, size=self.W[p][l].shape) * \
                               (np.random.random(self.W[p][l].shape) < self.mutation_prob)
                self.B[p][l] = self.B[p][l] + np.random.uniform(-self.mutation_rate, self.mutation_rate, size=self.B[p][l].shape) * \
                               (np.random.random(self.B[p][l].shape) < self.mutation_prob)

        elif self.mutation_type == 'forward_cycle':
            # Changes weights from first to last layer
            l = self.n
            for p in range(self.n_agents):
                self.W[p][l] = self.W[p][l] + np.random.uniform(-self.mutation_rate, self.mutation_rate, size=self.W[p][l].shape) * \
                               (np.random.random(self.W[p][l].shape) < self.mutation_prob)
                self.B[p][l] = self.B[p][l] + np.random.uniform(-self.mutation_rate, self.mutation_rate, size=self.B[p][l].shape) * \
                               (np.random.random(self.B[p][l].shape) < self.mutation_prob)
            l += 1  # forward
            self.n = l % (self.n_layers - 1)
        else:
            raise Exception('Mutation operation not implemented')

    def crossover(self):
        # Get best two agents
        idx_0, idx_1, *_ = np.argsort(self.cost)

        if self.crossover_type == 'none':
            # No crossover operation
            self.W = [np.copy(self.W[idx_0]) for p in range(self.n_agents)]
            self.B = [np.copy(self.B[idx_0]) for p in range(self.n_agents)]

        elif self.crossover_type == 'uniform':
            # Buffer weights of top two networks
            W_0_tmp, B_0_tmp = np.copy(self.W[idx_0]), np.copy(self.B[idx_0])
            W_1_tmp, B_1_tmp = np.copy(self.W[idx_1]), np.copy(self.B[idx_1])
            # Compute binary masks for crossover operation
            W_mask = [[np.random.randint(2, size=self.W[p][l].shape) for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]
            B_mask = [[np.random.randint(2, size=self.B[p][l].shape) for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]
            # Different uniform crossover for every offspring
            self.W = [[W_mask[p][l] * (W_0_tmp[l] - W_1_tmp[l]) + W_1_tmp[l] for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]
            self.B = [[B_mask[p][l] * (B_0_tmp[l] - B_1_tmp[l]) + B_1_tmp[l] for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]

        elif self.crossover_type == 'mean':
            # Buffer weights of top two networks
            W_0_tmp, B_0_tmp = np.copy(self.W[idx_0]), np.copy(self.B[idx_0])
            W_1_tmp, B_1_tmp = np.copy(self.W[idx_1]), np.copy(self.B[idx_1])
            # Pooled weights crossover
            self.W = [[0.5 * (W_0_tmp[l] + W_1_tmp[l]) for l in range(self.n_layers - 1)] for p in range(self.n_agents)]
            self.B = [[0.5 * (B_0_tmp[l] + B_1_tmp[l]) for l in range(self.n_layers - 1)] for p in range(self.n_agents)]

        elif self.crossover_type == 'neuron_wise':
            # Buffer weights of top two networks
            W_0_tmp, B_0_tmp = np.copy(self.W[idx_0]), np.copy(self.B[idx_0])
            W_1_tmp, B_1_tmp = np.copy(self.W[idx_1]), np.copy(self.B[idx_1])
            # Neuron-wise weight crossover
            mask = [[np.random.randint(2, size=self.B[p][l].shape) for l in range(self.n_layers - 1)] for p in
                    range(self.n_agents)]
            self.W = [[mask[p][l] * (W_0_tmp[l] - W_1_tmp[l]) + W_1_tmp[l] for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]
            self.B = [[mask[p][l] * (B_0_tmp[l] - B_1_tmp[l]) + B_1_tmp[l] for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]

        elif self.crossover_type == 'layer_wise':
            # Buffer weights of top two networks
            W_0_tmp, B_0_tmp = np.copy(self.W[idx_0]), np.copy(self.B[idx_0])
            W_1_tmp, B_1_tmp = np.copy(self.W[idx_1]), np.copy(self.B[idx_1])
            # Neuron-wise weight crossover
            self.W = [[W_0_tmp[l] if np.random.rand() < 0.5 else W_1_tmp[l] for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]
            self.B = [[B_0_tmp[l] if np.random.rand() < 0.5 else B_1_tmp[l] for l in range(self.n_layers - 1)] for p in
                      range(self.n_agents)]

        else:
            raise Exception('Crossover operation not implemented')

    def plot_progress_regression(self, epoch):
        marker_size = 1.0
        plt.plot(self.x_train_plt, self.y_train_plt, "bo", ms=marker_size, alpha=0.2)
        plt.plot(self.x_test_plt, self.y_test_plt, "ro", ms=marker_size, alpha=0.2)
        y_test_pred = self.prediction(self.best_agent, self.x_test)
        y_train_pred = self.prediction(self.best_agent, self.x_train)
        plt.plot(self.x_train, y_train_pred, "k-", alpha=0.7)
        plt.plot(self.x_test, y_test_pred, "k-", alpha=0.7)
        x_min, x_max = -1.1, 1.1
        y_min, y_max = -1.1, 1.1
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig(self.save_path + self.activation_fct + "_" + str(epoch) + ".png", dpi=150)
        plt.close()

    def plot_progress_classification(self, epoch, vis_type='prediction'):
        # Create validation grid of uniform distributed points for prediction landscape
        grid_resolution = 512
        x_min, x_max = -1.1, 1.1
        y_min, y_max = -1.1, 1.1
        xx_, yy_ = np.linspace(x_min, x_max, grid_resolution), np.linspace(y_min, y_max, grid_resolution)
        xx, yy = np.meshgrid(xx_, yy_)
        x = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
        y = self.prediction(self.best_agent, x)

        # custom discrete colormap (red, blue) and styles
        cmap = ListedColormap(np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]))

        # --- Create prediction landscape ---
        resolution = 512
        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        if vis_type == 'classification':
            y_pred = np.argmax(y, axis=1)
            zi = griddata((x[:, 0], x[:, 1]), y_pred, (xi[None, :], yi[:, None]), method='nearest')
            plt.contourf(xi, yi, zi, levels=1, alpha=0.4, cmap=cmap, vmin=0.0, vmax=1.0)
        elif vis_type == 'prediction':
            zi = griddata((x[:, 0], x[:, 1]), y[:, 0], (xi[None, :], yi[:, None]), method='cubic')
            plt.contourf(xi, yi, zi, levels=256, cmap='bwr', vmin=0.0, vmax=1.0)

        # --- Plot data points with scatter ---
        scatter_style = {'marker': 'o', 's': 10.0, 'alpha': 0.9, 'cmap': cmap, 'edgecolors': 'black', 'linewidths': 0.5}
        plt.scatter(self.x_train_plt[:, 0], self.x_train_plt[:, 1], c=self.y_train_plt[:, 1], **scatter_style)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        file_name = '{}_{}_{}.png'.format(vis_type, self.activation_fct, epoch)
        plt.tight_layout()
        plt.savefig(self.save_path + file_name, dpi=150)
        plt.close()

    def plot_dataset(self):
        # Custom discrete colormap (red, blue) and styles
        cmap = ListedColormap(np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]))
        # Parameters
        x_min, x_max = -1.1, 1.1
        y_min, y_max = -1.1, 1.1

        # --- Plot data points with scatter ---
        scatter_style = {'marker': 'o', 's': 10.0, 'alpha': 0.5, 'cmap': cmap, 'edgecolors': 'black', 'linewidths': 0.5}
        plt.scatter(self.x_train_plt[:, 0], self.x_train_plt[:, 1], c=self.y_train_plt[:, 1], **scatter_style)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig(self.save_path + 'problem.png', dpi=150)
        plt.close()
