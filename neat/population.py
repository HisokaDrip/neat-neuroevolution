import random
from dataclasses import dataclass, field

from .genome import Genome
from .innovation import InnovationTracker


@dataclass
class Species:
    key: int
    rep: Genome
    members: list = field(default_factory=list)
    best: float = float("-inf")
    last_improved: int = 0

    @property
    def peak(self):
        return max(m.fitness for m in self.members)


class Population:
    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.tracker = InnovationTracker(first_hidden_id=cfg.n_inputs + cfg.n_outputs)
        self._next_key = 0
        self.genomes = [Genome.minimal(self._key(), cfg, self.tracker, self.rng)
                        for _ in range(cfg.pop_size)]
        self.species = []
        self._next_species = 0
        self.generation = 0
        self.best = None
        self.history = []

    def _key(self):
        self._next_key += 1
        return self._next_key

    # --- speciation -----------------------------------------------------------

    def speciate(self):
        cfg = self.cfg
        for s in self.species:
            s.members = []
        for g in self.genomes:
            for s in self.species:
                if g.distance(s.rep, cfg) < cfg.compat_threshold:
                    s.members.append(g)
                    break
            else:
                s = Species(self._next_species, g, [g], last_improved=self.generation)
                self._next_species += 1
                self.species.append(s)
        self.species = [s for s in self.species if s.members]

        # nudge the threshold so we hover around target_species instead of
        # hand-tuning it per problem
        if len(self.species) < cfg.target_species:
            cfg.compat_threshold = max(0.3, cfg.compat_threshold - cfg.threshold_step)
        elif len(self.species) > cfg.target_species:
            cfg.compat_threshold += cfg.threshold_step

    # --- main loop ------------------------------------------------------------

    def run(self, evaluate, generations, target=None, on_generation=None):
        """evaluate(genomes) must set .fitness (> 0, higher is better) on each one."""
        for _ in range(generations):
            evaluate(self.genomes)
            gen_best = max(self.genomes, key=lambda g: g.fitness)
            if self.best is None or gen_best.fitness > self.best.fitness:
                self.best = gen_best.copy(gen_best.key)
                self.best.fitness = gen_best.fitness

            self.speciate()
            stats = self._stats(gen_best)
            self.history.append(stats)
            if on_generation:
                on_generation(stats)
            if target is not None and gen_best.fitness >= target:
                return self.best
            self._reproduce()
            self.generation += 1
        return self.best

    def _stats(self, gen_best):
        fits = [g.fitness for g in self.genomes]
        hidden, conns = gen_best.size
        return {
            "gen": self.generation,
            "best": gen_best.fitness,
            "mean": sum(fits) / len(fits),
            "species": len(self.species),
            "threshold": self.cfg.compat_threshold,
            "hidden": hidden,
            "conns": conns,
        }

    def _reproduce(self):
        cfg, rng = self.cfg, self.rng

        # drop species that haven't improved in a while, but never the top few
        for s in self.species:
            if s.peak > s.best:
                s.best, s.last_improved = s.peak, self.generation
        ranked = sorted(self.species, key=lambda s: s.best, reverse=True)
        keep = [s for i, s in enumerate(ranked)
                if i < cfg.protected_species
                or self.generation - s.last_improved < cfg.stagnation_limit]

        # explicit fitness sharing: a species' share is the mean fitness of its members
        shares = [sum(m.fitness for m in s.members) / len(s.members) for s in keep]
        total = sum(shares) or 1.0
        quotas = [max(1, round(cfg.pop_size * sh / total)) for sh in shares]
        # rounding drift: fix it on the biggest species
        drift = cfg.pop_size - sum(quotas)
        quotas[quotas.index(max(quotas))] += drift

        nxt = []
        for s, quota in zip(keep, quotas):
            if quota <= 0:
                continue
            members = sorted(s.members, key=lambda g: g.fitness, reverse=True)
            for elite in members[:min(cfg.elitism, quota)]:
                child = elite.copy(elite.key)
                child.fitness = None
                nxt.append(child)
            pool = members[:max(1, int(len(members) * cfg.survival_frac))]
            for _ in range(quota - min(cfg.elitism, quota)):
                if len(pool) == 1 or rng.random() < cfg.p_mutate_only:
                    child = rng.choice(pool).copy(self._key())
                else:
                    a, b = rng.sample(pool, 2)
                    if b.fitness > a.fitness:
                        a, b = b, a
                    child = Genome.crossover(self._key(), a, b, cfg, rng)
                child.mutate(cfg, self.tracker, rng)
                nxt.append(child)

        self.species = keep
        for s in self.species:
            # new representative: a random member of last generation (as in the paper)
            s.rep = rng.choice(s.members)
        self.genomes = nxt
