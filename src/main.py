from models.model import NeuralNetwork
from datetime import datetime


def main():
    problem_type = 'classification'
    problem = 'spirals' 
    n_points = 10000
    noise_level = 0.5
    activation_fct = 'tanh'
    n_outputs = 2
    n_inputs = 2

    #problem_type = 'regression'
    #problem = 'oscillation' 
    #n_points = 1000
    #noise_level = 0.1
    #activation_fct = 'relu'
    #n_outputs = 1
    #n_inputs = 1

    n_epochs = 100000
    n_agents = 4
    batch_size = 256 
    mutation_rate = 0.01
    mutation_prob = 0.1
    network_layers = (n_inputs,) + 2*(16,) + (n_outputs,)
    mutation_type = 'random_cycle' 
    crossover_type = 'neuron_wise' 
    save_path = './logs/{}/'.format(datetime.now().strftime("%Y%m%d%H%M%S"))

    network = NeuralNetwork(problem_type,
                            n_epochs,
                            problem,
                            n_points,
                            noise_level,
                            network_layers, 
                            n_outputs, 
                            n_agents, 
                            mutation_rate, 
                            mutation_prob,
                            mutation_type,
                            crossover_type,
                            batch_size,
                            activation_fct,
                            save_path)

    network.run()


if __name__ == '__main__':
    main()
