class InnovationTracker:
    """Hands out innovation numbers and hidden-node ids.

    The point of NEAT's historical markings is that the *same structural change*
    gets the *same number* no matter which genome stumbled on it. So we remember
    every (src, dst) connection we've ever numbered, and every connection split.
    Keeping it for the whole run (not just one generation like the paper) is a
    bit stricter and makes crossover line up even better.
    """

    def __init__(self, first_hidden_id):
        self._conn = {}
        self._split = {}
        self._next_innov = 0
        self._next_node = first_hidden_id

    def conn(self, src, dst):
        key = (src, dst)
        if key not in self._conn:
            self._conn[key] = self._next_innov
            self._next_innov += 1
        return self._conn[key]

    def split(self, src, dst):
        """Node id for splitting connection src->dst."""
        key = (src, dst)
        if key not in self._split:
            self._split[key] = self._next_node
            self._next_node += 1
        return self._split[key]
