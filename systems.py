from itertools import accumulate, repeat

class SimulationStep:
    def __init__(self, dynamics_model, measurement_model):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

    def sequence(self, n): return repeat(self, times=n)

    def propagate(self, x): return self.dynamics_model.simulate(x)
    def measure(self, x): return self.measurement_model.simulate(x)

    def run_simulation(self, x0, n):
        states = [x0]
        for k in range(n):
            states += [self.dynamics_model.simulate(states[-1])]

        measurements = [self.measurement_model.simulate(state) for state in states]
        return (states, measurements)
