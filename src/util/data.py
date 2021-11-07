import numpy as np
from sklearn import datasets

np.random.seed(42)


class DataLoader(object):
    def __init__(self, problem):
        self.problem = problem

    def get_data(self, n_points, noise_level):
        if self.problem == 'oscillation':
            x, y = self.make_oscillation(n_points, noise_level=noise_level)
        elif self.problem == 'moons':
            x, y = self.make_moons(n_points, noise_level=noise_level)
        elif self.problem == 'circles':
            x, y = self.make_circles(n_points, noise_level=noise_level)
        elif self.problem == 'checkerboard':
            x, y = self.make_checkerboard(n_points, noise_level=noise_level)
        elif self.problem == 'rectangles':
            x, y = self.make_rectangles(n_points, noise_level=noise_level)
        elif self.problem == 'spirals':
            x, y = self.make_spirals(n_points, noise_level=noise_level)
        else:
            raise Exception('Dataset not implemented.')

        return x, y

    @staticmethod
    def norm_data_2d(x):
        x[:, 0] = 2.0 * (x[:, 0]-x[:, 0].min()) / (x[:, 0].max()-x[:, 0].min()) - 1.0
        x[:, 1] = 2.0 * (x[:, 1]-x[:, 1].min()) / (x[:, 1].max()-x[:, 1].min()) - 1.0
        return x

    @staticmethod
    def norm_data_1d(x):
        return 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0

    def make_rectangles(self, n_points, noise_level=0.0, overlap_prop=0.5):
        n_half = n_points // 2
        # --- Create input data
        x0 = np.random.uniform(-1.0, 1.0, size=(n_half, 2))
        x1 = np.random.uniform(-1.0, 1.0, size=(n_half, 2))

        # Separate rectangles
        x0[:, 0] -= 1.0
        x1[:, 0] += 1.0

        # Merge data of rectangles
        x = np.concatenate([x0, x1], axis=0)
        y = 1*(x[:, 0] < 0.0)

        # Let rectangles overlap
        x[:n_half, 0] += overlap_prop
        x[n_half:, 0] -= overlap_prop

        mask = 1*(y == 1)
        y_one_hot = np.eye(2)[mask]

        if noise_level > 0.0:
            x = x + noise_level*np.random.normal(size=(n_points, 2))
        x = self.norm_data_2d(x)
        return x, y_one_hot

    def make_checkerboard(self, n_points, noise_level=1.0, n_xy_tiles=6):
        x = np.random.uniform(-(n_xy_tiles // 2) * np.pi, (n_xy_tiles // 2) * np.pi, size=(n_points, 2))
        mask = 1*(np.logical_or(np.logical_and(np.sin(x[:, 0]) > 0.0, np.sin(x[:, 1]) > 0.0),
                                np.logical_and(np.sin(x[:, 0]) < 0.0, np.sin(x[:, 1]) < 0.0)))
        y = np.eye(2)[mask]
        if noise_level > 0.0:
            x = x + noise_level*np.random.normal(size=(n_points, 2))
        x = self.norm_data_2d(x)
        return x, y

    def make_spirals(self, n_points, noise_level=0.0, compactness=2.5):
        n_half = n_points // 2
        r = np.sqrt(np.random.random(size=(n_half, 1))) * compactness * (2 * np.pi)
        r_1 = -np.cos(r) * r + np.random.normal(size=(n_half, 1)) * noise_level
        r_2 = np.sin(r) * r + np.random.normal(size=(n_half, 1)) * noise_level
        x = np.concatenate([np.concatenate([r_1, r_2], axis=-1), np.concatenate([-r_1, -r_2], axis=-1)], axis=0)
        y = np.concatenate([np.zeros(n_half), np.ones(n_half)], axis=0).astype(np.int)
        y_one_hot = np.eye(2)[y]
        x = self.norm_data_2d(x)
        return x, y_one_hot

    def make_circles(self, n_points, noise_level=0.0):
        x, y = datasets.make_circles(n_samples=n_points, factor=0.8, noise=noise_level)
        x = self.norm_data_2d(x)
        y = np.eye(2)[1*(y == 0)]
        return x, y

    def make_moons(self, n_points, noise_level=0.0):
        x, y = datasets.make_moons(n_samples=n_points, noise=noise_level)
        x = self.norm_data_2d(x)
        y = np.eye(2)[1*(y == 0)]
        return x, y

    def make_oscillation(self, n_points, noise=True, noise_level=0.0, ratio=0.95):
        f = lambda x: np.sin(4.0 * x)
        x1 = np.linspace(-5.0, -1.0, int(ratio*n_points)).reshape(-1, 1)
        x2 = np.linspace(1.0, 5.0, int((1.0 - ratio)*n_points)).reshape(-1, 1)
        x = np.concatenate((x1, x2))
        y = f(x)
        if noise:
            noise = noise_level * np.random.normal(0, 1, size=y.shape)
            noise = np.linspace(start=0.0, stop=1.0, num=len(y)).reshape(-1, 1) * noise
            y += noise
        x = self.norm_data_1d(x)
        y = self.norm_data_1d(y)
        return x, y

    # def make_oscillation(self, n_points, noise=True, noise_level=0.0, ratio=0.95):
    #     dx = 2.0 / float(n_points * (n_points + 1.0))
    #     n = 0.0
    #     x = np.zeros(shape=(n_points, 1))
    #     for i in range(n_points):
    #         x[i] = n * dx
    #         n += i
    #     x = 3.0 * 3.1415 * (2.0 * x - 1.0)
    #     f = lambda x: np.sin(3.0 * x)
    #     y = f(x)
    #     if noise:
    #         noise = noise_level * np.random.normal(0, 1, size=y.shape)
    #         noise = np.linspace(start=0.0, stop=1.0, num=len(y)).reshape(-1, 1) * noise
    #         y += noise
    #     x = self.norm_data_1d(x)
    #     y = self.norm_data_1d(y)
    #     return x, y
