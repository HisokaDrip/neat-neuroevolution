from .genome import INPUT, OUTPUT, steep_sigmoid


class FeedForwardNetwork:
    """Compiles a genome into a flat evaluation order.

    Only nodes that can actually influence an output are kept, so dangling
    hidden nodes (which NEAT produces a lot of) cost nothing at runtime.
    """

    def __init__(self, genome):
        nodes = genome.nodes
        self.inputs = sorted(n.id for n in nodes.values() if n.kind == INPUT)
        self.outputs = sorted(n.id for n in nodes.values() if n.kind == OUTPUT)
        live = [c for c in genome.conns.values() if c.enabled]

        # walk backwards from the outputs to find what matters
        incoming = {}
        for c in live:
            incoming.setdefault(c.dst, []).append(c)
        needed, stack = set(self.outputs), list(self.outputs)
        while stack:
            n = stack.pop()
            for c in incoming.get(n, ()):
                if c.src not in needed:
                    needed.add(c.src)
                    stack.append(c.src)

        # Kahn's algorithm over the needed subgraph
        compute = needed - set(self.inputs)
        deps = {n: [c for c in incoming.get(n, ()) if c.src in needed] for n in compute}
        ready = set(self.inputs)
        order, pending = [], set(compute)
        while pending:
            batch = [n for n in pending if all(c.src in ready for c in deps[n])]
            if not batch:
                raise ValueError("genome has a cycle; feed-forward only")
            for n in sorted(batch):
                order.append((n, nodes[n].bias, [(c.src, c.weight) for c in deps[n]]))
                ready.add(n)
                pending.discard(n)
        self.order = order

    def activate(self, xs):
        if len(xs) != len(self.inputs):
            raise ValueError(f"expected {len(self.inputs)} inputs, got {len(xs)}")
        vals = dict(zip(self.inputs, xs))
        for nid, bias, links in self.order:
            s = bias
            for src, w in links:
                s += vals[src] * w
            vals[nid] = steep_sigmoid(s)
        return [vals[o] for o in self.outputs]
