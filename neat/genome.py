import math
from dataclasses import dataclass, replace

INPUT, OUTPUT, HIDDEN = "input", "output", "hidden"


@dataclass
class NodeGene:
    id: int
    kind: str
    bias: float = 0.0


@dataclass
class ConnGene:
    innov: int
    src: int
    dst: int
    weight: float
    enabled: bool = True


def creates_cycle(conns, src, dst):
    """Would adding src->dst close a loop? (i.e. can dst already reach src)"""
    if src == dst:
        return True
    out = {}
    for c in conns:
        out.setdefault(c.src, []).append(c.dst)
    stack, seen = [dst], {dst}
    while stack:
        n = stack.pop()
        for m in out.get(n, ()):
            if m == src:
                return True
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False


class Genome:
    def __init__(self, key):
        self.key = key
        self.nodes = {}   # id -> NodeGene
        self.conns = {}   # innov -> ConnGene
        self.fitness = None

    @classmethod
    def minimal(cls, key, cfg, tracker, rng):
        """Inputs wired straight to outputs, no hidden layer. NEAT starts small on purpose."""
        g = cls(key)
        for i in range(cfg.n_inputs):
            g.nodes[i] = NodeGene(i, INPUT)
        for j in range(cfg.n_outputs):
            o = cfg.n_inputs + j
            g.nodes[o] = NodeGene(o, OUTPUT, rng.gauss(0, 1))
            for i in range(cfg.n_inputs):
                innov = tracker.conn(i, o)
                g.conns[innov] = ConnGene(innov, i, o, rng.gauss(0, 1))
        return g

    def copy(self, key):
        g = Genome(key)
        g.nodes = {k: replace(v) for k, v in self.nodes.items()}
        g.conns = {k: replace(v) for k, v in self.conns.items()}
        return g

    @property
    def size(self):
        hidden = sum(1 for n in self.nodes.values() if n.kind == HIDDEN)
        enabled = sum(1 for c in self.conns.values() if c.enabled)
        return hidden, enabled

    # --- mutation -----------------------------------------------------------

    def mutate(self, cfg, tracker, rng):
        if rng.random() < cfg.p_mutate_weights:
            self._mutate_weights(cfg, rng)
        if rng.random() < cfg.p_add_conn:
            self.add_connection(cfg, tracker, rng)
        if rng.random() < cfg.p_add_node:
            self.add_node(tracker, rng)
        if self.conns and rng.random() < cfg.p_toggle:
            c = rng.choice(list(self.conns.values()))
            c.enabled = not c.enabled

    def _mutate_weights(self, cfg, rng):
        lim = cfg.weight_range

        def nudge(v):
            if rng.random() < cfg.p_perturb:
                v += rng.gauss(0, cfg.perturb_sigma)
            else:
                v = rng.uniform(-lim, lim)
            return max(-lim, min(lim, v))

        for c in self.conns.values():
            c.weight = nudge(c.weight)
        for n in self.nodes.values():
            if n.kind != INPUT:
                n.bias = nudge(n.bias)

    def add_connection(self, cfg, tracker, rng, tries=20):
        srcs = [n for n in self.nodes.values() if n.kind != OUTPUT]
        dsts = [n for n in self.nodes.values() if n.kind != INPUT]
        existing = {(c.src, c.dst): c for c in self.conns.values()}
        for _ in range(tries):
            a, b = rng.choice(srcs).id, rng.choice(dsts).id
            if (a, b) in existing:
                c = existing[(a, b)]
                if not c.enabled:
                    # re-enabling counts as a structural mutation too
                    c.enabled = True
                    return True
                continue
            if creates_cycle(self.conns.values(), a, b):
                continue
            innov = tracker.conn(a, b)
            self.conns[innov] = ConnGene(innov, a, b, rng.uniform(-1, 1))
            return True
        return False

    def add_node(self, tracker, rng):
        live = [c for c in self.conns.values() if c.enabled]
        if not live:
            return False
        c = rng.choice(live)
        nid = tracker.split(c.src, c.dst)
        if nid in self.nodes:
            # this genome already split that link once before (then it got re-enabled)
            return False
        c.enabled = False
        self.nodes[nid] = NodeGene(nid, HIDDEN, 0.0)
        # weight 1 in, old weight out: the new node starts out (almost) behaving like the old link
        i1, i2 = tracker.conn(c.src, nid), tracker.conn(nid, c.dst)
        self.conns[i1] = ConnGene(i1, c.src, nid, 1.0)
        self.conns[i2] = ConnGene(i2, nid, c.dst, c.weight)
        return True

    # --- crossover / distance -------------------------------------------------

    @staticmethod
    def crossover(key, fit, weak, cfg, rng):
        """fit is the fitter parent; its disjoint/excess genes are the ones kept."""
        child = Genome(key)
        for innov, ca in fit.conns.items():
            cb = weak.conns.get(innov)
            gene = replace(ca if cb is None or rng.random() < 0.5 else cb)
            if cb is not None and (not ca.enabled or not cb.enabled):
                gene.enabled = rng.random() >= cfg.p_keep_disabled
            child.conns[innov] = gene
        for nid, na in fit.nodes.items():
            nb = weak.nodes.get(nid)
            child.nodes[nid] = replace(na if nb is None or rng.random() < 0.5 else nb)
        return child

    def distance(self, other, cfg):
        a, b = self.conns, other.conns
        if not a and not b:
            return 0.0
        shared = a.keys() & b.keys()
        disjoint = len(a) + len(b) - 2 * len(shared)
        w = sum(abs(a[k].weight - b[k].weight) for k in shared) / len(shared) if shared else 0.0
        n = max(len(a), len(b))
        n = 1 if n < 20 else n
        return cfg.c_disjoint * disjoint / n + cfg.c_weight * w

    # --- io -------------------------------------------------------------------

    def to_dict(self):
        return {
            "key": self.key,
            "fitness": self.fitness,
            "nodes": [vars(n) for n in self.nodes.values()],
            "conns": [vars(c) for c in self.conns.values()],
        }

    @classmethod
    def from_dict(cls, d):
        g = cls(d["key"])
        g.fitness = d.get("fitness")
        g.nodes = {n["id"]: NodeGene(**n) for n in d["nodes"]}
        g.conns = {c["innov"]: ConnGene(**c) for c in d["conns"]}
        return g

    def describe(self):
        hidden, enabled = self.size
        lines = [f"genome {self.key}: {hidden} hidden, {enabled} enabled conns"]
        for c in sorted(self.conns.values(), key=lambda c: c.innov):
            if c.enabled:
                lines.append(f"  {c.src:>3} -> {c.dst:<3} w={c.weight:+.3f}")
        return "\n".join(lines)


def steep_sigmoid(x):
    # the 4.9 slope from Stanley & Miikkulainen; near-linear around 0, saturates fast
    x = max(-60.0, min(60.0, 4.9 * x))
    return 1.0 / (1.0 + math.exp(-x))
