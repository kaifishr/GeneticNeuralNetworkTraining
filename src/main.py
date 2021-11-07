from models.model import NeuralNetwork


def main():
    config = dict()

    # Task
    config["task"] = "classification"  # classification, regression

    # Data
    config["problem"] = "moons"  # Regression: oscillation, Classification: moons, circles, checkerboard, rectangles, spirals
    config["n_points"] = 4000
    config["noise_level"] = 0.05

    # Network
    config["n_dims_input"] = 2
    config["n_dims_hidden"] = 8
    config["n_dims_output"] = 2
    config["n_dims_sampling_dist"] = 128
    config["n_hidden"] = 2

    # Training
    config["n_epochs"] = 99999999
    config["batch_size"] = 256
    config["n_agents"] = 4
    config["mutation_rate"] = 1.0e-02
    config["mutation_prob"] = 1.0e-01

    # Stats
    config["stats_every_n_epochs"] = 200
    config["plots_every_n_epochs"] = 5000

    network = NeuralNetwork(config)
    network.run()


if __name__ == '__main__':
    main()
