import random

import pytest

from neat import Config, FeedForwardNetwork, Genome, InnovationTracker
from neat.genome import HIDDEN, ConnGene, creates_cycle


@pytest.fixture
def setup():
    cfg = Config(n_inputs=3, n_outputs=2)
    tracker = InnovationTracker(first_hidden_id=5)
    return cfg, tracker, random.Random(0)


def test_minimal_genome_is_fully_connected(setup):
    cfg, tracker, rng = setup
    g = Genome.minimal(1, cfg, tracker, rng)
    assert len(g.conns) == 6
    assert {(c.src, c.dst) for c in g.conns.values()} == {(i, o) for i in range(3) for o in (3, 4)}


def test_same_structure_gets_same_innovation(setup):
    cfg, tracker, rng = setup
    a = Genome.minimal(1, cfg, tracker, rng)
    b = Genome.minimal(2, cfg, tracker, rng)
    assert a.conns.keys() == b.conns.keys()
    assert tracker.split(0, 3) == tracker.split(0, 3)
    assert tracker.split(0, 3) != tracker.split(1, 3)


def test_add_node_preserves_behaviour_roughly(setup):
    cfg, tracker, rng = setup
    g = Genome.minimal(1, cfg, tracker, rng)
    before = len(g.conns)
    assert g.add_node(tracker, rng)
    assert len(g.conns) == before + 2
    disabled = [c for c in g.conns.values() if not c.enabled]
    assert len(disabled) == 1
    hidden = [n for n in g.nodes.values() if n.kind == HIDDEN]
    assert len(hidden) == 1
    into = [c for c in g.conns.values() if c.dst == hidden[0].id]
    out = [c for c in g.conns.values() if c.src == hidden[0].id]
    assert into[0].weight == 1.0 and out[0].weight == disabled[0].weight


def test_creates_cycle():
    conns = [ConnGene(0, 0, 5, 1.0), ConnGene(1, 5, 6, 1.0), ConnGene(2, 6, 3, 1.0)]
    assert creates_cycle(conns, 6, 5)
    assert creates_cycle(conns, 3, 0)
    assert creates_cycle(conns, 5, 5)
    assert not creates_cycle(conns, 0, 6)


def test_mutation_never_breaks_feedforward(setup):
    cfg, tracker, rng = setup
    cfg.p_add_conn, cfg.p_add_node = 0.6, 0.3
    g = Genome.minimal(1, cfg, tracker, rng)
    for _ in range(300):
        g.mutate(cfg, tracker, rng)
        FeedForwardNetwork(g).activate([0.1, -0.2, 0.3])  # raises on a cycle
    assert g.size[0] > 5


def test_crossover_takes_disjoint_genes_from_fitter_parent(setup):
    cfg, tracker, rng = setup
    a = Genome.minimal(1, cfg, tracker, rng)
    b = a.copy(2)
    a.add_node(tracker, rng)
    b.add_connection(cfg, tracker, rng) or b.add_node(tracker, rng)
    child = Genome.crossover(3, a, b, cfg, rng)
    assert child.conns.keys() == a.conns.keys()
    assert child.nodes.keys() == a.nodes.keys()


def test_distance(setup):
    cfg, tracker, rng = setup
    a = Genome.minimal(1, cfg, tracker, rng)
    assert a.distance(a, cfg) == 0
    b = a.copy(2)
    for c in b.conns.values():
        c.weight += 1.0
    assert b.distance(a, cfg) == pytest.approx(cfg.c_weight * 1.0)
    b.add_node(tracker, rng)
    assert b.distance(a, cfg) == pytest.approx(cfg.c_weight * 1.0 + 2 * cfg.c_disjoint)


def test_roundtrip_dict(setup):
    cfg, tracker, rng = setup
    g = Genome.minimal(1, cfg, tracker, rng)
    g.add_node(tracker, rng)
    g.fitness = 3.5
    h = Genome.from_dict(g.to_dict())
    x = [0.3, 0.6, -0.9]
    assert FeedForwardNetwork(h).activate(x) == FeedForwardNetwork(g).activate(x)
    assert h.fitness == 3.5
