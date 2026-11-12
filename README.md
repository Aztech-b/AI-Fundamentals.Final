# Campus Delivery Robot with Obstacle Awareness

A campus courier must deliver a parcel while a corridor may be temporarily blocked and its sensor may be wrong. This Python console simulation combines **A* search, Bayesian inference, and a one-step Markov decision process (MDP)** to complete that delivery. The robot can move, wait for better conditions, or follow a newly planned detour.

The three algorithms share a delivery goal: A* selects the next corridor, Bayes estimates its blockage probability, and the MDP uses that estimate to decide whether to enter. Waiting or a failed move triggers A* again with the updated belief.

The original assignment documents are retained unchanged in [the project brief](<docs/requirements/Final Project(1).docx>) and [the detailed guidelines](<docs/requirements/MiniAI_FPTopics(1).docx>).

## Run

Use Python **3.10 or later**. There are no third-party dependencies. Development and verification used Python 3.14.7.

From the project folder:

```bash
python3 main.py
```

The default runs all four scenarios. To demonstrate one at a time:

```bash
python3 main.py --scenario clear
python3 main.py --scenario temporary
python3 main.py --scenario detour
python3 main.py --scenario unreachable
```

Each run prints the map, A* expansion scores, Bayesian calculations, MDP action values, executed actions, and final metrics. `python3 main.py --help` lists the options. An expected `unreachable` scenario is a demonstration result and does not cause a process error.

## Test

```bash
python3 -m unittest discover -s tests -v
```

The 30 tests cover hand-calculated algorithm results, route optimality on small fixtures, probability validation, repeated observations, action ties, complete deliveries, false alerts, missed obstacles, hidden-state separation, and finite termination. No installation or internet connection is needed.

## Environment and delivery loop

Coordinates use `(row, column)`, starting at zero. The normal 5x5 map is:

```text
. . . . .
. . # . .
S . ? . G
. . # . .
. . . . .
```

`S=(2,0)` is pickup, `G=(2,4)` is the destination, `#` is a known wall, and `?=(2,2)` is the uncertain corridor. During execution `R` marks the robot and `*` marks the remaining route. The direct route takes four moves; an alternative around the corridor takes eight.

1. A* plans using known walls and the current blockage belief.
2. The robot follows known-clear route cells normally.
3. Before entering the uncertain corridor, it obtains a sensor reading and updates its belief using Bayes.
4. The MDP compares Move with Wait using the updated belief.
5. One tick passes. Beliefs are forecast forward, and the simulator executes the action against the actual next-tick condition.
6. A blocked attempt leaves the robot in place and establishes that the corridor is blocked. Successful entry establishes that it is clear. Waiting or a blocked attempt triggers replanning; otherwise the robot follows the committed route.

The actual obstacle timeline belongs to the simulator. A*, Bayes, and the MDP receive known map information and beliefs; they never receive the hidden timeline. Sensor readings refer to tick `t`; an attempted entry is checked at `t+1`, since the corridor can change while an action takes place.

Delivery ends with `delivered`, `unreachable` through the known map, or `step_limit` after 100 actions. A time limit means the robot could not finish within its budget; it does not prove that the destination is impossible to reach.

## The three algorithms

### A* search

**Input:** known walls, current position, destination, and blockage probabilities.

Entering a cell costs `1 + 8*p`, where `p` is its blockage probability. Known-clear cells have `p=0`. The search prioritizes `f=g+h`, where `g` is accumulated weighted cost and `h` is Manhattan distance. The heuristic remains admissible because each step costs at least one. Equal priorities use a stable insertion order for repeatable output.

**Output:** a least-cost path, its weighted cost, and expanded cells with `g/h/f` values. An empty path indicates that known walls prevent a route.

**Why it fits:** the robot needs a route that accounts for both distance and estimated blockage. The risk weight of 8 is an illustrative design choice. Weighted route cost is a planning score, distinct from actual distance and elapsed ticks.

Initially, the direct path costs `4 + 8*0.20 = 5.6`, less than the eight-move detour. After an alert and one wait, the estimated risk can make the detour cheaper. A* therefore changes the robot's actual route.

### Bayesian inference

**Input:** a prior blockage probability and one noisy reading, `alert` or `clear`.

The initial prior is `P(Blocked)=0.20`. The assumed sensor model is:

| Reading | P(reading given Blocked) | P(reading given Clear) |
|---|---:|---:|
| alert | 0.80 | 0.10 |
| clear | 0.20 | 0.90 |

```text
P(Blocked | reading) = prior * P(reading | Blocked)
                       / (prior * P(reading | Blocked)
                          + (1-prior) * P(reading | Clear))
```

**Output:** the posterior blockage probability, used by both the MDP and subsequent route planning.

For example, an alert with prior `0.20` gives `0.16 / (0.16+0.08) = 2/3`. A clear reading with the same prior gives `1/19`.

Beliefs also evolve between readings: `p_next = 0.10*(1-p) + 0.80*p`. The first reading in the standard demo occurs after one movement tick, so its prior is `0.24`, not `0.20`. For the clear scenario, the resulting posterior is approximately `0.065574`.

**Why it fits:** a sensor alert is evidence rather than certainty. Repeated readings refine the estimate while the transition model accounts for possible changes over time. Readings are assumed conditionally independent given the current corridor state.

### One-step MDP

**Input:** the posterior belief over Clear/Blocked, a transition model, and rewards.

For both actions, the corridor follows this assumed transition table:

| Current state | Next Clear | Next Blocked |
|---|---:|---:|
| Clear | 0.90 | 0.10 |
| Blocked | 0.20 | 0.80 |

Actions do not control the corridor's evolution. Waiting gives conditions time to change without attempting entry. Rewards depend on the action and resulting state:

| Action | Next Clear | Next Blocked |
|---|---:|---:|
| Move | +10 | -20 |
| Wait | -1 | -1 |

```text
Q(b,a) = sum_current b(current)
         * sum_next T(current,next) * R(a,next)

Q(Move) = 7 - 21*p
Q(Wait) = -1
```

**Output:** the greater-valued action and both action values. Ties within floating-point tolerance choose Wait. For `p=2/3`, Move has value `-7` and Wait has value `-1`, so the robot waits.

**Why it fits:** the robot must choose an action with uncertain consequences and different rewards. This is the one-step expected-reward MDP allowed by the detailed project guidelines. Its state values after that step are zero; it does not compute a globally optimal delivery policy or solve a full partially observable MDP. Repeated execution supplies the complete delivery behavior.

## Results and evaluation

Running `python3 main.py` produces this summary:

```text
Delivery evaluation
Scenario       Status       Moves Waits Blocked Ticks Replans
clear          delivered        4     0       0     4       0
temporary      delivered        4     4       0     8       4
detour         delivered        8     1       0     9       1
unreachable    unreachable      0     0       0     0       0
```

- **Clear:** uses the four-move direct route with no waiting or blocked attempts.
- **Temporary:** only the center row is traversable, and the corridor clears at tick 4. The robot waits four times, including one extra wait while evidence still suggests sufficient risk, then completes delivery.
- **Detour:** the corridor stays blocked. The robot waits once after an alert, replans, and completes the eight-move alternative without entering the corridor.
- **Unreachable:** additional known walls isolate the destination. A* reports no route before any movement occurs.

Moves count successful entries; Waits count waiting actions; Blocked counts unsuccessful attempted entries. Ticks equal Moves + Waits + Blocked. Replans count A* searches after the initial search, including searches that return the same route.

Noise tests expose the tradeoffs. One false alert in the otherwise clear scenario increases delivery from four ticks to nine by causing a wait and unnecessary detour. A missed alert in the blocked scenario produces one blocked attempt, followed by replanning and successful delivery in nine ticks. These are deterministic examples, not estimates of real-world accuracy or proof that this controller always improves delivery time.

## Live demonstration and oral defense

Run the scenarios in the order `clear`, `detour`, `temporary`, `unreachable`. Point to the initial route, sensor calculation, two action values, executed behavior, changed route, and final evaluation table. Then run the tests.

Be ready to explain:

| Question | Explanation |
|---|---|
| How do the algorithms work together? | A* chooses the corridor; Bayes assesses it; the MDP controls entry; new beliefs affect the next A* search. |
| Why does A* find the least-cost route? | It uses accumulated edge costs and an admissible Manhattan heuristic. |
| Why can an alert be wrong? | The assumed sensor has a 10% alert likelihood even when the corridor is clear. |
| Why does the prior change before a new observation? | The corridor may clear or become blocked as time passes, modeled by the transition table. |
| Why wait after a clear reading? | One reading may not overcome the accumulated evidence of blockage. |
| Where are transitions used in the MDP? | Both state probabilities and transition probabilities weight the rewards of resulting states. |
| Why can the robot make a blocked attempt? | Noisy observations and changing conditions allow an incorrect decision; the simulator prevents movement through the blockage. |
| What are the limitations? | The map is small, probabilities are assumed, and one-step decisions can be locally sensible but globally inefficient. |

Every team member should be able to explain all three algorithms and the shared delivery loop.

## Assumptions and future development

Only the marked corridor is uncertain; other traversable cells are known clear. Obstacle timelines and sensor readings are scripted for repeatability. The probabilistic tables describe the robot's assumed model; the demonstrations are selected possible observations and events, not random samples or learned parameters. Sensor overrides in tests demonstrate false alerts and missed obstacles.

Possible improvements include calibrating probabilities with observations, longer-horizon decisions, multiple uncertain corridors, and a real campus map with multiple deliveries. The console simulation is the current deliverable.

## Code and AI assistance

`main.py` contains the three algorithm implementations, scenario fixtures, delivery loop, and console interface. `tests/test_delivery.py` contains the automated checks. `docs/requirements/` contains the original assignment documents.

The implementation, tests, and documentation were produced with assistance from OpenAI Codex. No course-practice source code was supplied; these are standalone standard-library implementations. Review the calculations, run the scenarios and tests, and understand the code before the oral defense. Record actual team contributions when preparing a team submission.
