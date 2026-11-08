"""Run XOR and double-pole over a bunch of seeds and print the numbers I quote in the README."""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.cartpole import double_pole, episode  # noqa: E402
from examples.pole_balance import MAX_STEPS, VERIFY_STEPS, controller  # noqa: E402
from examples.xor import solved, xor_fitness  # noqa: E402
from neat import Config, Population  # noqa: E402


def summarize(name, evals, wins, total):
    print(f"{name:<12} solved {wins}/{total}   evals median {statistics.median(evals):.0f}  "
          f"mean {statistics.mean(evals):.0f}  min {min(evals)}  max {max(evals)}")


def main(seeds=range(1, 21)):
    seeds = list(seeds)
    evals, wins, hidden = [], 0, []
    for s in seeds:
        cfg = Config(n_inputs=2, n_outputs=1)
        pop = Population(cfg, seed=s)
        best = pop.run(xor_fitness, 300, target=15.9)
        evals.append((pop.generation + 1) * cfg.pop_size)
        wins += solved(best)
        hidden.append(best.size[0])
    summarize("xor", evals, wins, len(seeds))
    print(f"{'':<12} hidden nodes in winners: median {statistics.median(hidden):.0f}")

    evals, wins = [], 0
    env = double_pole()
    for s in seeds:
        cfg = Config(n_inputs=6, n_outputs=1)
        pop = Population(cfg, seed=s)

        def fitness(genomes):
            for g in genomes:
                g.fitness = float(episode(controller(g), env, MAX_STEPS))

        best = pop.run(fitness, 200, target=MAX_STEPS)
        evals.append((pop.generation + 1) * cfg.pop_size)
        wins += episode(controller(best), double_pole(), VERIFY_STEPS) >= VERIFY_STEPS
    summarize("double pole", evals, wins, len(seeds))
    print(f"{'':<12} ('solved' here = survives the 100k-step verification run)")


if __name__ == "__main__":
    main()
