import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from util.data import DataLoader

np.random.seed(19937)


class NeuralNetwork(object):

    def __init__(self, config):

        self.task = config["task"]
        self.problem = config["problem"]
        self.n_points = config["n_points"]
        self.noise_level = config["noise_level"]
        self.n_dims_input = config["n_dims_input"]
        self.n_dims_hidden = config["n_dims_hidden"]
        self.n_dims_output = config["n_dims_output"]
        self.n_dims_sampling_dist = config["n_dims_sampling_dist"]
        self.n_hidden = config["n_hidden"]
        self.n_epochs = config["n_epochs"]
        self.n_agents = config["n_agents"]
        self.batch_size = config["batch_size"]
        self.mutation_rate = config["mutation_rate"]
        self.mutation_prob = config["mutation_prob"]
        self.plots_every_n_epochs = config["plots_every_n_epochs"]
        self.stats_every_n_epochs = config["stats_every_n_epochs"]

        network_layers = [self.n_dims_input] + [self.n_dims_hidden for _ in range(self.n_hidden)] + [self.n_dims_output]

        self.n_layers = len(network_layers)

        # Generate data
        data_loader = DataLoader(problem=self.problem)
        self.x_train, self.y_train = data_loader.get_data(n_points=self.n_points, noise_level=self.noise_level)
        self.x_test, self.y_test = data_loader.get_data(n_points=400, noise_level=self.noise_level)

        # Initialize weights
        self.W = self._init_weights(network_layers)
        self.B = self._init_biases(network_layers)

        # Other parameters
        self.idx_best = 0

    def _init_weights(self, network_layers):
        return [[np.random.normal(size=(network_layers[l+1], network_layers[l], self.n_dims_sampling_dist)) * np.sqrt(2.0 / network_layers[l])
                 for l in range(self.n_layers-1)] for _ in range(self.n_agents)]

    def _init_biases(self, network_layers):
        return [[np.zeros((network_layers[l + 1], self.n_dims_sampling_dist))
                 for l in range(self.n_layers - 1)] for _ in range(self.n_agents)]

    @staticmethod
    def _relu_mean(x_mean):
        return x_mean * (x_mean > 0.0)

    @staticmethod
    def _relu_var(x_mean, x_var):
        return x_var * (x_mean > 0.0)

    @staticmethod
    def _tanh_mean(x_mean):
        return np.tanh(x_mean)

    @staticmethod
    def _tanh_var(x_mean, x_var):
        return x_var * (1.0 - np.tanh(x_mean)**2)**2

    @staticmethod
    def _linear_mean(x_mean, w_mean, b_mean):
        return np.dot(x_mean, w_mean.T) + b_mean

    @staticmethod
    def _linear_var(x_mean, x_var, w_mean, w_var, b_var):
        term_1 = np.dot(x_var, w_var.T)
        term_2 = np.dot(x_var, np.square(w_mean).T)
        term_3 = np.dot(np.square(x_mean), w_var.T)
        term_4 = b_var
        return term_1 + term_2 + term_3 + term_4

    def _grouped_rand_idx(self):
        idx = np.random.permutation(len(self.x_train))
        return [idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size)]

    def forward(self, p, x):
        for w, b in zip(self.W[p][:-1], self.B[p][:-1]):
            w = w.mean(axis=-1)
            b = b.mean(axis=-1)
            x = np.dot(x, w.T) + b
            x = self._relu_mean(x)
        w = self.W[p][-1].mean(axis=-1)
        b = self.B[p][-1].mean(axis=-1)
        y_pred = self._tanh_mean(np.dot(x, w.T) + b)
        return y_pred

    @staticmethod
    def _comp_mean_var(param):
        return param.mean(axis=-1), param.var(axis=-1)

    def forward_mean_var(self, p, x):
        x_mean = x
        x_var = np.zeros_like(x)

        for w, b in zip(self.W[p][:-1], self.B[p][:-1]):
            # Compute mean and variance of sampling distributions
            w_mean, w_var = self._comp_mean_var(w)
            b_mean, b_var = self._comp_mean_var(b)
            # Variance and mean propagation linear
            x_var = self._linear_var(x_mean, x_var, w_mean, w_var, b_var)
            x_mean = self._linear_mean(x_mean, w_mean, b_mean)
            # Variance and mean propagation activation function
            x_var = self._relu_var(x_mean, x_var)
            x_mean = self._relu_mean(x_mean)

        # Compute mean and variance of sampling distributions
        w_mean, w_var = self._comp_mean_var(self.W[p][-1])
        b_mean, b_var = self._comp_mean_var(self.B[p][-1])
        # Variance and mean propagation linear
        x_var = self._linear_var(x_mean, x_var, w_mean, w_var, b_var)
        x_mean = self._linear_mean(x_mean, w_mean, b_mean)
        # Variance and mean propagation activation function
        # x_var = self._tanh_var(x_mean, x_var)
        x_mean = self._tanh_mean(x_mean)

        return x_mean, x_var # / x_var.max()

    def _comp_stats(self, x_data, y_data):
        y_pred = self.forward(self.idx_best, x_data)
        loss = self._comp_loss(y_data, y_pred)
        accuracy = self._comp_accuracy(y_data, y_pred)
        return loss, accuracy

    @staticmethod
    def _comp_loss(y_data, y_pred):
        """Computes average loss per instance.
        """
        return np.sum((y_data - y_pred) ** 2) / len(y_data)

    @staticmethod
    def _comp_accuracy(y_data, y_pred):
        """Computes accuracy for classification task.
        """
        return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_data, axis=1)) / len(y_data)

    def _plot2d(self, grid_resolution=200):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Plot ground truth
        x_min = self.x_train.min(axis=0, keepdims=True)
        x_max = self.x_train.max(axis=0, keepdims=True)
        x_train = grid_resolution * (self.x_train - x_min) / (x_max - x_min)
        axes[0].scatter(x_train[:, 0], x_train[:, 1], c=self.y_train[:, 0], s=1.0, cmap="bwr")
        axes[1].scatter(x_train[:, 0], x_train[:, 1], c=self.y_train[:, 0], s=1.0, cmap="bwr")

        # Plot prediction
        domain = 1.0
        x_min, x_max = -domain, domain
        y_min, y_max = -domain, domain
        xx_, yy_ = np.linspace(x_min, x_max, grid_resolution), np.linspace(y_min, y_max, grid_resolution)
        xx, yy = np.meshgrid(xx_, yy_)
        x = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
        y_mean, y_var = self.forward_mean_var(self.idx_best, x)
        axes[0].imshow(y_mean[:, 0].reshape(grid_resolution, grid_resolution), cmap="magma")
        axes[1].imshow(y_var.mean(axis=-1).reshape(grid_resolution, grid_resolution), cmap="magma")

        # fig.tight_layout(pad=0)

        for ax in axes:
            ax.set_axis_off()

        return fig

    def _plot1d(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Plot ground truth
        ax.plot(self.x_train, self.y_train, "r.", markersize=0.4, lw=0.5, alpha=0.75)

        # Plot prediction
        n_samples_test = 2000
        x = np.linspace(start=-1.25, stop=1.25, num=n_samples_test).reshape(n_samples_test, 1)
        y_mean, y_var = self.forward_mean_var(self.idx_best, x)

        ax.plot(x, y_mean, "g-", lw=1.0, alpha=1.0)
        ax.fill_between(x[:, 0], y_mean[:, 0] - y_var[:, 0], y_mean[:, 0] + y_var[:, 0], color="green", alpha=0.2)

        return fig

    def run(self):
        writer = SummaryWriter()

        for epoch in tqdm(range(1, self.n_epochs+1)):

            for idx in self._grouped_rand_idx():
                x_data, y_data = self.x_train[idx], self.y_train[idx]

                batch_loss = list()

                for p in range(self.n_agents):
                    y_pred = self.forward(p, x_data)
                    batch_loss.append(self._comp_loss(y_data, y_pred))

                self.idx_best = np.argmin(batch_loss)
                self._clone_agent()
                self._mutate_agent()
                self._clip_agent()

            if epoch % self.stats_every_n_epochs == 0:
                test_loss, test_accuracy = self._comp_stats(self.x_test, self.y_test)
                writer.add_scalar("loss", test_loss, epoch)
                writer.add_scalar("accuracy", test_accuracy, epoch)

            if epoch % self.plots_every_n_epochs == 0:
                # writer.add_figure("plot", self._plot1d(), epoch)
                writer.add_figure("plot", self._plot2d(), epoch)

    def _clone_agent(self):
        """Clone best agent."""
        self.W = self._clone(self.W)
        self.B = self._clone(self.B)

    def _clone(self, params):
        return [[param for param in params[self.idx_best]] for _ in range(self.n_agents)]

    def _mutate_agent(self):
        """Apply mutation to agents."""
        self.W = self._mutate(self.W)
        self.B = self._mutate(self.B)

    def _mutate(self, params):
        return [[p + np.random.uniform(-self.mutation_rate, self.mutation_rate, size=p.shape)
                 * (np.random.random(p.shape) < self.mutation_prob) for p in param] for param in params]

    def _clip_agent(self):
        """Clips agent's parameters."""
        self.W = self._clip(self.W)
        self.B = self._clip(self.B)

    @staticmethod
    def _clip(params, a_min=-10.0, a_max=10.0):
        return [[p.clip(a_min, a_max) for p in param] for param in params]

