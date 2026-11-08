# neat-neuroevolution

NEAT (NeuroEvolution of Augmenting Topologies) written from scratch in plain Python. No numpy, no torch, no neat-python. Just the standard library.

NEAT evolves the weights *and* the shape of a neural network at the same time. Every network starts as inputs wired straight to outputs, and the genetic algorithm adds hidden nodes and connections only when they help. I wanted to really understand the three tricks from the [Stanley & Miikkulainen 2002 paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) that make it work, so here they are, all in about 500 lines:

- **Historical markings.** Every structural mutation gets an innovation number, so two totally different topologies can still be lined up gene by gene for crossover (`neat/innovation.py`).
- **Speciation.** Genomes are grouped by a compatibility distance, so a brand new hidden node gets a few generations to tune its weights before it has to compete with the established networks. The threshold adjusts itself to keep roughly 10 species around, so I don't have to hand-tune it per problem.
- **Starting minimal.** Topology only grows when it pays for itself, so the solutions stay small.

I tested it on the two classic NEAT benchmarks: XOR and double pole balancing.

## Running it

Nothing to install for the examples. Python 3.9+ is enough.

```bash
python examples/xor.py
python examples/pole_balance.py --poles 2               # evolve, verify, save runs/double_best.json
python examples/pole_balance.py --poles 2 --watch runs/double_best.json
python examples/benchmark.py                            # 20 seeds of each, takes about a minute

pip install pytest && python -m pytest -q
```

## XOR

XOR is the sanity check because it can't be solved without at least one hidden node, so NEAT has to discover structure on its own:

```
gen   0  best  8.894  mean  5.112  species  1  hidden 0  conns 2
gen  10  best  9.000  mean  5.742  species 14  hidden 1  conns 3
gen  20  best 15.618  mean  6.793  species 10  hidden 2  conns 7
gen  30  best 15.886  mean  6.916  species 11  hidden 2  conns 7
gen  40  best 15.886  mean  7.543  species 16  hidden 2  conns 7

stopped at generation 41, best fitness 15.9374
  (0.0, 0.0) -> 0.0344  (want 0)
  (0.0, 1.0) -> 0.9968  (want 1)
  (1.0, 0.0) -> 0.9668  (want 1)
  (1.0, 1.0) -> 0.0744  (want 0)
solved
```

## Double pole balancing

Two poles stand on one cart: a 1 m pole and a 10 cm pole. The controller pushes the cart left or right and has to keep both poles upright and the cart on a 4.8 m track. The short pole reacts about 10x faster than the long one, which is what makes this hard. I used the Wieland equations, the same ones the paper used, with all velocities given as inputs (the "Markovian" version).

During evolution a controller has to last 10,000 steps (100 simulated seconds). The winner then gets a verification run of 100,000 steps, which is about 16 minutes of balancing.

```
gen   0  best     174 steps  mean    24.6  species  1  hidden 0  conns 6
...
gen  17  best     305 steps  mean    81.2  species 10  hidden 1  conns 9
gen  18  best     698 steps  mean    94.9  species 10  hidden 3  conns 13
gen  19  best   10000 steps  mean   157.8  species 10  hidden 3  conns 13

3000 evaluations, 0.8s wall clock
verification run: balanced for 1000 simulated seconds (pass)
```

Watching it in the terminal:

```
|                                       #                     | t=  2.96s  x=+0.79  poles  -1.21°  -1.90°
```

## Numbers over 20 seeds

From `examples/benchmark.py`:

```
xor          solved 20/20   evals median 6225  mean 6532  min 3150  max 11550
             hidden nodes in winners: median 3
double pole  solved 18/20   evals median 7050  mean 7508  min 2700  max 15900
             ('solved' here = survives the 100k-step verification run)
```

The paper reports about 4,755 evaluations for XOR and 3,600 for Markovian double pole. My setup isn't identical (Euler integration, different mutation rates, a self-adjusting threshold), so I'm happy being in the same ballpark. The 2 double pole runs that "failed" did hit 100 s during evolution, they just drifted off the track somewhere before 1000 s.

## Layout

```
neat/genome.py       node/connection genes, mutations, crossover, compatibility distance
neat/innovation.py   innovation numbers + split-node ids
neat/network.py      compiles a genome into a pruned topological evaluation order
neat/population.py   speciation, stagnation, fitness sharing, reproduction
envs/cartpole.py     single/double cart-pole physics + an ascii renderer
examples/            xor, pole balancing, benchmark
```

## Bug I hit

My first double pole run was "solved" at generation 0 by a random network. It turned out the sign convention in those equations expects `g = -9.8`. With `+9.8` the poles hang *down* like pendulums and any controller passes. There's a test for that now (`test_poles_fall_without_control`).

## What I'd do next

- Non-Markovian double pole (no velocity inputs). That needs recurrent connections, so the network compiler would need to handle cycles.
- HyperNEAT, or at least CPPN output, so it can evolve patterns and not just controllers.
- Parallel fitness evaluation with `multiprocessing`. Evaluations are embarrassingly parallel and that's where all the time goes.
