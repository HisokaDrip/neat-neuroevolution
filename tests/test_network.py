import pytest

from neat import FeedForwardNetwork, Genome
from neat.genome import HIDDEN, INPUT, OUTPUT, ConnGene, NodeGene, steep_sigmoid


def handmade():
    g = Genome(0)
    g.nodes = {0: NodeGene(0, INPUT), 1: NodeGene(1, INPUT),
               2: NodeGene(2, OUTPUT, 0.1), 3: NodeGene(3, HIDDEN, -0.2),
               4: NodeGene(4, HIDDEN, 0.0)}
    g.conns = {
        0: ConnGene(0, 0, 3, 0.5), 1: ConnGene(1, 1, 3, -1.0),
        2: ConnGene(2, 3, 2, 2.0), 3: ConnGene(3, 0, 2, 0.7),
        4: ConnGene(4, 1, 4, 1.0),  # dead end: 4 never reaches an output
        5: ConnGene(5, 1, 2, 9.0, enabled=False),
    }
    return g


def test_matches_hand_computation():
    net = FeedForwardNetwork(handmade())
    x0, x1 = 0.4, 0.9
    h = steep_sigmoid(-0.2 + 0.5 * x0 - 1.0 * x1)
    want = steep_sigmoid(0.1 + 2.0 * h + 0.7 * x0)
    assert net.activate([x0, x1])[0] == pytest.approx(want)


def test_prunes_dead_nodes_and_disabled_links():
    net = FeedForwardNetwork(handmade())
    assert [n for n, _, _ in net.order] == [3, 2]
    links_into_out = dict(net.order[1][2])
    assert 1 not in links_into_out


def test_output_with_no_inputs_is_just_bias():
    g = Genome(0)
    g.nodes = {0: NodeGene(0, INPUT), 1: NodeGene(1, OUTPUT, 0.0)}
    assert FeedForwardNetwork(g).activate([123.0]) == [0.5]


def test_wrong_input_count():
    with pytest.raises(ValueError):
        FeedForwardNetwork(handmade()).activate([1.0])
