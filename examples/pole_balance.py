"""Evolve a controller for single or double pole balancing (with velocities).

    python examples/pole_balance.py --poles 2
    python examples/pole_balance.py --poles 2 --watch runs/double_best.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.cartpole import double_pole, episode, render, single_pole  # noqa: E402
from neat import Config, FeedForwardNetwork, Genome, Population  # noqa: E402

MAX_STEPS = 10_000      # 100 simulated seconds during evolution
VERIFY_STEPS = 100_000  # the paper's bar: survive 1000s (~16 min) once evolved


def make_env(poles):
    return single_pole() if poles == 1 else double_pole()


def controller(genome):
    net = FeedForwardNetwork(genome)
    return lambda obs: net.activate(obs)[0]


def watch(path, poles, seconds):
    genome = Genome.from_dict(json.loads(Path(path).read_text()))
    env, policy = make_env(poles), controller(genome)
    obs = env.reset()
    frames = int(seconds / 0.01)
    for i in range(frames):
        if not env.step(policy(obs)):
            print("\n... and it fell over")
            break
        obs = env.observe()
        if i % 5 == 0:
            print("\r" + render(env), end="", flush=True)
            time.sleep(0.02)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poles", type=int, choices=(1, 2), default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--watch", metavar="GENOME_JSON")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    if args.watch:
        watch(args.watch, args.poles, args.seconds)
        return

    env = make_env(args.poles)

    def fitness(genomes):
        for g in genomes:
            g.fitness = float(episode(controller(g), env, MAX_STEPS))

    cfg = Config(n_inputs=2 + 2 * args.poles, n_outputs=1)
    pop = Population(cfg, seed=args.seed)
    t0 = time.time()

    def log(s):
        print(f"gen {s['gen']:>3}  best {s['best']:>7.0f} steps  mean {s['mean']:>7.1f}  "
              f"species {s['species']:>2}  hidden {s['hidden']}  conns {s['conns']}")

    best = pop.run(fitness, args.generations, target=MAX_STEPS, on_generation=log)
    evals = (pop.generation + 1) * cfg.pop_size
    print(f"\n{evals} evaluations, {time.time() - t0:.1f}s wall clock")
    print(best.describe())

    survived = episode(controller(best), make_env(args.poles), VERIFY_STEPS)
    print(f"verification run: balanced for {survived * 0.01:.0f} simulated seconds "
          f"({'pass' if survived >= VERIFY_STEPS else 'fail'})")

    out = Path(args.out or f"runs/{'single' if args.poles == 1 else 'double'}_best.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(best.to_dict(), indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
