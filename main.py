"""Campus parcel delivery using A*, Bayesian inference, and a one-step MDP."""

import argparse
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf, isfinite
import sys
from typing import TextIO


Cell = tuple[int, int]
SIZE = 5
START = (2, 0)
GOAL = (2, 4)
CORRIDOR = (2, 2)
WALLS = frozenset({(1, 2), (3, 2)})
PRIOR = 0.20
RISK_WEIGHT = 8
SENSOR = {"alert": (0.80, 0.10), "clear": (0.20, 0.90)}
# Rows and columns are [Clear, Blocked]; the corridor evolves under either action.
TRANSITION = ((0.90, 0.10), (0.20, 0.80))
REWARDS = {"Move": (10, -20), "Wait": (-1, -1)}


def check_probability(value: float) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probabilities must be finite numbers between 0 and 1.")


@dataclass
class SearchResult:
    path: list[Cell]
    cost: float
    expanded: list[tuple[Cell, float, int]]


def astar(
    size: int,
    walls: frozenset[Cell],
    start: Cell,
    goal: Cell,
    risks: dict[Cell, float],
) -> SearchResult:
    """Find the least-cost route; entering a cell costs 1 + 8 * P(blocked)."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Grid size must be a positive integer.")
    for row, col in {*walls, *risks, start, goal}:
        if not (0 <= row < size and 0 <= col < size):
            raise ValueError("All coordinates must be inside the grid.")
    if start in walls or goal in walls:
        raise ValueError("Start and goal must be traversable cells.")
    for probability in risks.values():
        check_probability(probability)

    def heuristic(cell: Cell) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    order = count()
    frontier = [(heuristic(start), next(order), 0.0, start)]
    best = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    expanded = []
    while frontier:
        _, _, cost, cell = heappop(frontier)
        if cost != best[cell]:
            continue
        expanded.append((cell, cost, heuristic(cell)))
        if cell == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return SearchResult(list(reversed(path)), cost, expanded)

        # Stable tie-breaking makes demonstrations repeatable: right, up, down, left.
        for dr, dc in ((0, 1), (-1, 0), (1, 0), (0, -1)):
            neighbor = (cell[0] + dr, cell[1] + dc)
            row, col = neighbor
            if not (0 <= row < size and 0 <= col < size) or neighbor in walls:
                continue
            candidate = cost + 1 + RISK_WEIGHT * risks.get(neighbor, 0)
            if candidate < best.get(neighbor, inf):
                best[neighbor] = candidate
                parent[neighbor] = cell
                heappush(frontier, (candidate + heuristic(neighbor), next(order), candidate, neighbor))
    return SearchResult([], inf, expanded)


def bayes_update(prior: float, likelihood_blocked: float, likelihood_clear: float) -> float:
    """P(blocked | observation), from a prior and the two sensor likelihoods."""
    for probability in (prior, likelihood_blocked, likelihood_clear):
        check_probability(probability)
    evidence = prior * likelihood_blocked + (1 - prior) * likelihood_clear
    if evidence == 0:
        raise ValueError("The observation has zero probability under this model.")
    return prior * likelihood_blocked / evidence


def predict_blockage(probability: float) -> float:
    """Advance the corridor belief by one tick without inspecting the actual world."""
    check_probability(probability)
    return (1 - probability) * TRANSITION[0][1] + probability * TRANSITION[1][1]


def mdp_decision(probability: float) -> tuple[str, dict[str, float]]:
    """One-step Q(b,a) = sum_s b(s) sum_next T(s,next) R(a,next)."""
    check_probability(probability)
    belief = (1 - probability, probability)
    values = {
        action: sum(
            belief[state] * TRANSITION[state][next_state] * rewards[next_state]
            for state in range(2)
            for next_state in range(2)
        )
        for action, rewards in REWARDS.items()
    }
    # Treat floating-point near-ties as ties, with Wait as the conservative choice.
    action = "Move" if values["Move"] - values["Wait"] > 1e-12 else "Wait"
    return action, values


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    walls: frozenset[Cell]
    blocked_until: int | None = 0
    sensor_overrides: tuple[tuple[int, str], ...] = ()

    def blocked_at(self, tick: int) -> frozenset[Cell]:
        """Simulator truth: None means blocked throughout the episode."""
        if self.blocked_until is None or tick < self.blocked_until:
            return frozenset({CORRIDOR})
        return frozenset()

    def observe(self, cell: Cell, tick: int) -> str:
        if cell != CORRIDOR:
            raise ValueError("Only the uncertain corridor has a noisy sensor.")
        signal = "alert" if cell in self.blocked_at(tick) else "clear"
        for override_tick, override_signal in self.sensor_overrides:
            if override_tick == tick:
                signal = override_signal
                break
        if signal not in SENSOR:
            raise ValueError("Sensor readings must be 'alert' or 'clear'.")
        return signal


SCENARIOS = {
    "clear": Scenario("clear", "Deliver through the clear central corridor.", WALLS),
    "temporary": Scenario(
        "temporary",
        "The only passage is blocked until tick 4; wait for evidence that it clears.",
        frozenset((row, col) for row in range(SIZE) for col in range(SIZE) if row != 2),
        blocked_until=4,
    ),
    "detour": Scenario(
        "detour", "The central corridor stays blocked; find a route around it.", WALLS,
        blocked_until=None,
    ),
    "unreachable": Scenario(
        "unreachable", "Known walls isolate the destination.",
        WALLS | {(1, 4), (2, 3), (3, 4)},
    ),
}


@dataclass
class Step:
    tick: int
    position: Cell
    target: Cell
    observation: str | None
    prior: float
    posterior: float
    action: str
    values: dict[str, float]
    blocked: bool


@dataclass
class DeliveryResult:
    scenario: str
    status: str
    route_taken: list[Cell]
    steps: list[Step]
    searches: list[SearchResult]

    @property
    def moves(self) -> int:
        return len(self.route_taken) - 1

    @property
    def waits(self) -> int:
        return sum(step.action == "Wait" for step in self.steps)

    @property
    def blocked_attempts(self) -> int:
        return sum(step.blocked for step in self.steps)

    @property
    def replans(self) -> int:
        return max(0, len(self.searches) - 1)


def render_map(walls: frozenset[Cell], route: list[Cell], position: Cell) -> str:
    """Render the robot's known map, never the simulator's hidden blockages."""
    symbols = {cell: "*" for cell in route}
    symbols.update({cell: "#" for cell in walls})
    symbols.update({START: "S", GOAL: "G", CORRIDOR: "?", position: "R"})
    return "\n".join(" ".join(symbols.get((row, col), ".") for col in range(SIZE)) for row in range(SIZE))


def print_search(search: SearchResult, start: Cell, output: TextIO) -> None:
    print(f"\nA* input: {start} -> {GOAL}; step cost = 1 + 8*p; h = Manhattan distance", file=output)
    print("  Expanded cell       g      h      f", file=output)
    for cell, cost, heuristic in search.expanded:
        print(f"  {str(cell):<16} {cost:6.3f} {heuristic:6.3f} {cost + heuristic:6.3f}", file=output)
    if search.path:
        print(f"  Output: {search.path}; {len(search.path) - 1} moves; cost={search.cost:.3f}", file=output)
    else:
        print("  Output: no route through the known map.", file=output)
    print("  A* finds the least-cost route, including the estimated cost of blockage.", file=output)


def print_step(step: Step, position: Cell, output: TextIO) -> None:
    print(f"\nTick {step.tick}: robot={step.position}, next cell={step.target}", file=output)
    if step.observation is not None:
        blocked_likelihood, clear_likelihood = SENSOR[step.observation]
        print(f"  Bayes input: prior={step.prior:.6f}, signal={step.observation}", file=output)
        print(
            f"  P(blocked|signal) = ({step.prior:.6f}*{blocked_likelihood:.2f}) / "
            f"({step.prior:.6f}*{blocked_likelihood:.2f} + "
            f"{1 - step.prior:.6f}*{clear_likelihood:.2f}) = {step.posterior:.6f}",
            file=output,
        )
        print("  Bayes turns a noisy reading into a blockage probability.", file=output)
        print(f"  MDP input belief [Clear, Blocked]: [{1 - step.posterior:.6f}, {step.posterior:.6f}]", file=output)
        print("  Q(b,a) = sum_s b(s) * sum_next T(s,next) * R(a,next)", file=output)
        for action, value in step.values.items():
            print(f"  Q({action}) = {value:.6f}", file=output)
        print(f"  MDP output: {step.action}; choose the action with greater expected one-step reward.", file=output)
    else:
        print("  Known-clear route cell: Move.", file=output)
    if step.blocked:
        print("  Outcome: blocked attempt; robot stays in place; contact sets P(blocked)=1.", file=output)
    elif step.action == "Wait":
        print("  Outcome: waited one tick; request a new route using the updated belief.", file=output)
    else:
        print(f"  Outcome: moved to {position}.", file=output)


def simulate(scenario: Scenario, max_steps: int = 100, output: TextIO | None = None) -> DeliveryResult:
    """Run delivery; algorithm inputs contain beliefs and known walls, not world truth."""
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("Maximum steps must be a nonnegative integer.")
    beliefs = {CORRIDOR: PRIOR}
    position = START
    route: list[Cell] = []
    needs_plan = True
    result = DeliveryResult(scenario.name, "step_limit", [START], [], [])
    if output is not None:
        print(f"\n=== {scenario.name}: {scenario.description} ===", file=output)
        print(render_map(scenario.walls, [], position), file=output)
        print("R=robot S=pickup G=destination #=known wall ?=uncertain corridor *=route", file=output)

    while len(result.steps) < max_steps:
        if needs_plan:
            search = astar(SIZE, scenario.walls, position, GOAL, beliefs)
            result.searches.append(search)
            if output is not None:
                print_search(search, position, output)
            if not search.path:
                result.status = "unreachable"
                break
            route = search.path
            needs_plan = False

        tick = len(result.steps)
        target = route[1]
        before = position
        observation = None
        prior = posterior = 0.0
        values = {}
        action = "Move"
        if target in beliefs:
            prior = beliefs[target]
            observation = scenario.observe(target, tick)
            posterior = bayes_update(prior, *SENSOR[observation])
            beliefs[target] = posterior
            action, values = mdp_decision(posterior)

        # The world advances between sensing and the action outcome; forecast all beliefs too.
        beliefs = {cell: predict_blockage(probability) for cell, probability in beliefs.items()}
        blocked = action == "Move" and target in scenario.blocked_at(tick + 1)
        if blocked:
            beliefs[target] = 1.0
            needs_plan = True
        elif action == "Wait":
            needs_plan = True
        else:
            position = target
            result.route_taken.append(position)
            route = route[1:]
            if target in beliefs:
                beliefs[target] = 0.0

        step = Step(tick, before, target, observation, prior, posterior, action, values, blocked)
        result.steps.append(step)
        if output is not None:
            print_step(step, position, output)
            print(render_map(scenario.walls, route, position), file=output)
        if position == GOAL:
            result.status = "delivered"
            break

    if output is not None:
        print(
            f"\nResult: {result.status}; moves={result.moves}, waits={result.waits}, "
            f"blocked attempts={result.blocked_attempts}, ticks={len(result.steps)}, "
            f"replans={result.replans}", file=output,
        )
    return result


def print_summary(results: list[DeliveryResult], output: TextIO) -> None:
    print("\nDelivery evaluation", file=output)
    print(f"{'Scenario':<14} {'Status':<12} {'Moves':>5} {'Waits':>5} {'Blocked':>7} {'Ticks':>5} {'Replans':>7}", file=output)
    for result in results:
        print(
            f"{result.scenario:<14} {result.status:<12} {result.moves:>5} {result.waits:>5} "
            f"{result.blocked_attempts:>7} {len(result.steps):>5} {result.replans:>7}",
            file=output,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all", help="delivery scenario (default: all)")
    args = parser.parse_args()
    selected = SCENARIOS.values() if args.scenario == "all" else [SCENARIOS[args.scenario]]
    print("Campus Delivery Robot: A* + Bayes + one-step MDP")
    print("Illustrative model; deterministic obstacle timelines and sensor readings.")
    print(f"MDP T [Clear, Blocked], for either action: {TRANSITION}")
    print(f"Rewards by resulting state: {REWARDS}")
    print("Sense at tick t; world advances; execute at t+1. Belief forecast: p_next=0.1+0.7*p.")
    results = [simulate(scenario, output=sys.stdout) for scenario in selected]
    print_summary(results, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
