"""Evolve XOR. The classic NEAT sanity check: it can't be solved without a hidden node."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neat import Config, FeedForwardNetwork, Population  # noqa: E402

CASES = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]


def xor_fitness(genomes):
    for g in genomes:
        net = FeedForwardNetwork(g)
        err = sum((net.activate(x)[0] - y) ** 2 for x, y in CASES)
        g.fitness = (4.0 - err) ** 2   # max 16


def solved(genome):
    net = FeedForwardNetwork(genome)
    return all(round(net.activate(x)[0]) == y for x, y in CASES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--generations", type=int, default=300)
    args = ap.parse_args()

    # no explicit bias input: every non-input node carries its own evolvable bias
    cfg = Config(n_inputs=2, n_outputs=1)
    pop = Population(cfg, seed=args.seed)

    def log(s):
        if s["gen"] % 10 == 0:
            print(f"gen {s['gen']:>3}  best {s['best']:6.3f}  mean {s['mean']:6.3f}  "
                  f"species {s['species']:>2}  hidden {s['hidden']}  conns {s['conns']}")

    best = pop.run(xor_fitness, args.generations, target=15.9, on_generation=log)
    print(f"\nstopped at generation {pop.generation}, best fitness {best.fitness:.4f}")
    print(best.describe())
    net = FeedForwardNetwork(best)
    for x, y in CASES:
        print(f"  {x} -> {net.activate(x)[0]:.4f}  (want {y:.0f})")
    print("solved" if solved(best) else "not solved")


if __name__ == "__main__":
    main()
