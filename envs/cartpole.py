"""Cart-pole physics, single and double pole.

Equations are the ones from Wieland (1991), which is what the NEAT paper used
for its double pole benchmark. Integration is plain Euler at 10ms -- RK4 is
nicer, but Euler is what most of the old literature did and it's 4x cheaper,
which matters when you simulate ~10^7 steps in pure Python.
"""
import math

# negative on purpose: with the sign convention in these equations a positive g
# makes the poles hang *down*, which is a very easy problem (ask me how I know)
G = -9.8
CART_MASS = 1.0
TRACK = 2.4          # |x| beyond this = off the track
FORCE = 10.0
DT = 0.01


class CartPole:
    """N poles hinged on one cart. poles = [(half_length, mass), ...]"""

    def __init__(self, poles=((0.5, 0.1),), max_angle_deg=36.0, start_angle_deg=1.0):
        self.poles = poles
        self.max_angle = math.radians(max_angle_deg)
        self.start_angle = math.radians(start_angle_deg)
        self.reset()

    def reset(self):
        self.x = 0.0
        self.dx = 0.0
        # only the long pole starts tilted; everything else upright
        self.th = [self.start_angle] + [0.0] * (len(self.poles) - 1)
        self.dth = [0.0] * len(self.poles)
        self.t = 0
        return self.observe()

    def observe(self):
        """Scaled to roughly [-1, 1] so evolved weights don't have to fight units."""
        obs = [self.x / TRACK, self.dx / 2.0]
        for th, dth in zip(self.th, self.dth):
            obs += [th / self.max_angle, dth / 4.0]
        return obs

    def step(self, action):
        """action in [0, 1]; 0.5 = no push. Returns True while still balanced."""
        force = FORCE * (2.0 * action - 1.0)
        eff_force, eff_mass = 0.0, 0.0
        for (l, m), th, dth in zip(self.poles, self.th, self.dth):
            s, c = math.sin(th), math.cos(th)
            eff_force += m * l * dth * dth * s + 0.75 * m * c * G * s
            eff_mass += m * (1.0 - 0.75 * c * c)
        ddx = (force + eff_force) / (CART_MASS + eff_mass)
        for i, (l, m) in enumerate(self.poles):
            th = self.th[i]
            ddth = -0.75 / l * (ddx * math.cos(th) + G * math.sin(th))
            self.th[i] += DT * self.dth[i]
            self.dth[i] += DT * ddth
        self.x += DT * self.dx
        self.dx += DT * ddx
        self.t += 1
        return self.alive()

    def alive(self):
        return abs(self.x) <= TRACK and all(abs(t) <= self.max_angle for t in self.th)


def single_pole():
    return CartPole(poles=((0.5, 0.1),), max_angle_deg=12.0, start_angle_deg=3.0)


def double_pole():
    # long pole 1m, short pole 0.1m -- the short one is the nasty part
    return CartPole(poles=((0.5, 0.1), (0.05, 0.01)), max_angle_deg=36.0, start_angle_deg=1.0)


def episode(policy, env, max_steps):
    """Run policy(obs) -> action until the poles fall or time is up. Returns steps survived."""
    obs = env.reset()
    for _ in range(max_steps):
        if not env.step(policy(obs)):
            break
        obs = env.observe()
    return env.t


def render(env, width=61):
    """One-line ascii frame. Good enough to watch it wobble in a terminal."""
    row = [" "] * width
    pos = int((env.x / TRACK + 1) / 2 * (width - 1))
    pos = max(0, min(width - 1, pos))
    row[pos] = "#"
    tilts = " ".join(f"{math.degrees(t):+6.2f}°" for t in env.th)
    return f"|{''.join(row)}| t={env.t * DT:6.2f}s  x={env.x:+.2f}  poles {tilts}"
