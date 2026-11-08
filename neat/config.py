from dataclasses import dataclass


@dataclass
class Config:
    n_inputs: int
    n_outputs: int
    pop_size: int = 150

    # compatibility distance: d = c1 * (E + D) / N + c3 * W
    # (the paper splits excess/disjoint into c1/c2 but uses the same value for both)
    c_disjoint: float = 1.0
    c_weight: float = 0.4
    compat_threshold: float = 3.0
    target_species: int = 10
    threshold_step: float = 0.3

    # mutation
    p_mutate_weights: float = 0.8
    p_perturb: float = 0.9          # otherwise the weight is replaced outright
    perturb_sigma: float = 0.5
    weight_range: float = 4.0
    p_add_conn: float = 0.08
    p_add_node: float = 0.03
    p_toggle: float = 0.01
    p_keep_disabled: float = 0.75

    # reproduction
    survival_frac: float = 0.4
    p_mutate_only: float = 0.25
    elitism: int = 1
    stagnation_limit: int = 15
    protected_species: int = 2
