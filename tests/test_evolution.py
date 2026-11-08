from envs.cartpole import double_pole, episode, single_pole
from examples.pole_balance import controller
from examples.xor import solved, xor_fitness
from neat import Config, Population


def test_xor_gets_solved():
    cfg = Config(n_inputs=2, n_outputs=1)
    pop = Population(cfg, seed=2)
    best = pop.run(xor_fitness, 300, target=15.9)
    assert solved(best)
    assert best.size[0] >= 1, "xor needs at least one hidden node"


def test_same_seed_same_run():
    def go():
        pop = Population(Config(n_inputs=2, n_outputs=1), seed=11)
        pop.run(xor_fitness, 15)
        return [h["best"] for h in pop.history]
    assert go() == go()


def test_population_size_stays_fixed():
    cfg = Config(n_inputs=2, n_outputs=1, pop_size=60)
    pop = Population(cfg, seed=3)
    sizes = []
    pop.run(xor_fitness, 25, on_generation=lambda s: sizes.append(len(pop.genomes)))
    assert set(sizes) == {60}


def test_poles_fall_without_control():
    # guards against the sign bug where gravity makes the poles hang down
    assert episode(lambda obs: 0.5, single_pole(), 10_000) < 200
    assert episode(lambda obs: 0.5, double_pole(), 10_000) < 200


def test_double_pole_gets_balanced():
    env = double_pole()

    def fitness(genomes):
        for g in genomes:
            g.fitness = float(episode(controller(g), env, 10_000))

    pop = Population(Config(n_inputs=6, n_outputs=1), seed=7)
    best = pop.run(fitness, 100, target=10_000)
    assert best.fitness >= 10_000
