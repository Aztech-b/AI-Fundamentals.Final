"""Algorithm calculations and complete delivery behavior, using only unittest."""

from dataclasses import replace
from io import StringIO
from pathlib import Path
import subprocess
import sys
import unittest

from main import (
    CORRIDOR, GOAL, PRIOR, SCENARIOS, SENSOR, SIZE, START, TRANSITION, WALLS,
    astar, bayes_update, mdp_decision, predict_blockage, simulate,
)


class AStarTests(unittest.TestCase):
    def test_direct_route_and_visible_search_steps(self):
        result = astar(SIZE, WALLS, START, GOAL, {CORRIDOR: PRIOR})
        self.assertEqual(result.path, [(2, col) for col in range(SIZE)])
        self.assertAlmostEqual(result.cost, 5.6)
        self.assertEqual(result.expanded[0], (START, 0, 4))
        self.assertEqual(result.expanded[-1][0], GOAL)

    def test_high_risk_selects_eight_move_detour(self):
        result = astar(SIZE, WALLS, START, GOAL, {CORRIDOR: 2 / 3})
        self.assertEqual(len(result.path) - 1, 8)
        self.assertAlmostEqual(result.cost, 8)
        self.assertNotIn(CORRIDOR, result.path)

    def test_weighted_cost_is_optimal_on_hand_calculated_map(self):
        # The direct route costs 2+8*p; either alternative costs 4.
        for probability, expected_cost, expected_moves in [(0.1, 2.8, 2), (0.3, 4, 4)]:
            with self.subTest(probability=probability):
                result = astar(3, frozenset(), (1, 0), (1, 2), {(1, 1): probability})
                self.assertAlmostEqual(result.cost, expected_cost)
                self.assertEqual(len(result.path) - 1, expected_moves)

    def test_unreachable_goal(self):
        result = astar(SIZE, SCENARIOS["unreachable"].walls, START, GOAL, {})
        self.assertEqual(result.path, [])
        self.assertEqual(result.cost, float("inf"))

    def test_start_is_already_goal(self):
        result = astar(SIZE, WALLS, START, START, {})
        self.assertEqual(result.path, [START])
        self.assertEqual(result.cost, 0)

    def test_invalid_map_inputs(self):
        for size, walls, start, goal in [
            (0, frozenset(), START, GOAL),
            (5, WALLS, (-1, 0), GOAL),
            (5, WALLS, START, (5, 4)),
            (5, WALLS | {START}, START, GOAL),
            (5, WALLS | {GOAL}, START, GOAL),
            (5, WALLS | {(6, 0)}, START, GOAL),
        ]:
            with self.subTest(size=size, walls=walls, start=start, goal=goal):
                with self.assertRaises(ValueError):
                    astar(size, walls, start, goal, {})
        for invalid in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(risk=invalid):
                with self.assertRaises(ValueError):
                    astar(SIZE, WALLS, START, GOAL, {CORRIDOR: invalid})


class BayesianTests(unittest.TestCase):
    def test_worked_sensor_updates(self):
        self.assertAlmostEqual(bayes_update(PRIOR, *SENSOR["alert"]), 2 / 3)
        self.assertAlmostEqual(bayes_update(PRIOR, *SENSOR["clear"]), 1 / 19)

    def test_repeated_evidence_uses_predicted_belief(self):
        posterior = bayes_update(PRIOR, *SENSOR["alert"])
        next_prior = predict_blockage(posterior)
        self.assertAlmostEqual(next_prior, 17 / 30)
        self.assertAlmostEqual(bayes_update(next_prior, *SENSOR["alert"]), 136 / 149)
        self.assertAlmostEqual(bayes_update(next_prior, *SENSOR["clear"]), 34 / 151)

    def test_time_prediction(self):
        for prior, expected in [(0, 0.1), (1, 0.8), (0.2, 0.24)]:
            with self.subTest(prior=prior):
                self.assertAlmostEqual(predict_blockage(prior), expected)

    def test_certain_prior_with_possible_evidence(self):
        self.assertEqual(bayes_update(0, 0.8, 0.1), 0)
        self.assertEqual(bayes_update(1, 0.8, 0.1), 1)

    def test_impossible_observation(self):
        for values in [(0.5, 0, 0), (0, 1, 0), (1, 0, 1)]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "zero probability"):
                    bayes_update(*values)

    def test_invalid_probabilities(self):
        for invalid in [-0.1, 1.1, float("nan"), float("inf")]:
            for index in range(3):
                values = [0.2, 0.8, 0.1]
                values[index] = invalid
                with self.subTest(values=values):
                    with self.assertRaises(ValueError):
                        bayes_update(*values)
            with self.assertRaises(ValueError):
                predict_blockage(invalid)


class MDPTests(unittest.TestCase):
    def test_transition_rows_are_probability_distributions(self):
        self.assertEqual(len(TRANSITION), 2)
        for row in TRANSITION:
            self.assertEqual(len(row), 2)
            self.assertAlmostEqual(sum(row), 1)
            self.assertTrue(all(0 <= probability <= 1 for probability in row))

    def test_clear_evidence_recommends_move(self):
        action, values = mdp_decision(1 / 19)
        self.assertEqual(action, "Move")
        self.assertAlmostEqual(values["Move"], 112 / 19)
        self.assertAlmostEqual(values["Wait"], -1)

    def test_alert_recommends_wait(self):
        action, values = mdp_decision(2 / 3)
        self.assertEqual(action, "Wait")
        self.assertAlmostEqual(values["Move"], -7)
        self.assertAlmostEqual(values["Wait"], -1)

    def test_rewards_include_state_transitions(self):
        self.assertAlmostEqual(mdp_decision(0)[1]["Move"], 7)
        self.assertAlmostEqual(mdp_decision(1)[1]["Move"], -14)

    def test_ties_wait_and_nearby_beliefs_choose_correctly(self):
        self.assertEqual(mdp_decision(8 / 21)[0], "Wait")
        self.assertEqual(mdp_decision(8 / 21 - 1e-6)[0], "Move")
        self.assertEqual(mdp_decision(8 / 21 + 1e-6)[0], "Wait")

    def test_invalid_belief(self):
        for invalid in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(belief=invalid):
                with self.assertRaises(ValueError):
                    mdp_decision(invalid)


class DeliveryTests(unittest.TestCase):
    def test_clear_delivery_takes_four_moves(self):
        result = simulate(SCENARIOS["clear"])
        self.assertEqual(result.status, "delivered")
        self.assertEqual(result.route_taken, [(2, col) for col in range(SIZE)])
        self.assertEqual((result.moves, result.waits, result.blocked_attempts), (4, 0, 0))
        self.assertAlmostEqual(result.steps[1].prior, 0.24)
        self.assertEqual(result.steps[1].action, "Move")

    def test_temporary_obstacle_waits_then_completes(self):
        scenario = SCENARIOS["temporary"]
        result = simulate(scenario)
        self.assertEqual(result.status, "delivered")
        self.assertEqual((result.moves, result.waits, result.blocked_attempts), (4, 4, 0))
        self.assertEqual(len(result.steps), 8)
        self.assertEqual(result.steps[4].observation, "clear")
        self.assertEqual(result.steps[4].action, "Wait")
        corridor_entry = next(step for step in result.steps if step.target == CORRIDOR and step.action == "Move")
        self.assertGreaterEqual(corridor_entry.tick + 1, scenario.blocked_until)
        self.assertEqual(result.route_taken[-1], GOAL)

    def test_persistent_obstacle_triggers_wait_and_detour(self):
        result = simulate(SCENARIOS["detour"])
        self.assertEqual(result.status, "delivered")
        self.assertEqual((result.moves, result.waits, result.blocked_attempts), (8, 1, 0))
        self.assertEqual((len(result.steps), result.replans), (9, 1))
        self.assertIn(CORRIDOR, result.searches[0].path)
        self.assertNotIn(CORRIDOR, result.searches[1].path)
        self.assertNotIn(CORRIDOR, result.route_taken)

    def test_known_unreachable_goal_stops_before_moving(self):
        result = simulate(SCENARIOS["unreachable"])
        self.assertEqual(result.status, "unreachable")
        self.assertEqual(result.route_taken, [START])
        self.assertEqual(result.steps, [])

    def test_false_alert_has_a_measurable_cost(self):
        noisy = replace(SCENARIOS["clear"], sensor_overrides=((1, "alert"),))
        result = simulate(noisy)
        self.assertEqual(result.status, "delivered")
        self.assertEqual((result.moves, result.waits, result.blocked_attempts), (8, 1, 0))
        self.assertNotIn(CORRIDOR, noisy.blocked_at(1))
        self.assertEqual(result.steps[1].action, "Wait")

    def test_missed_obstacle_cannot_move_robot_through_blockage(self):
        missed = replace(SCENARIOS["detour"], sensor_overrides=((1, "clear"),))
        result = simulate(missed)
        self.assertEqual(result.status, "delivered")
        self.assertEqual(result.blocked_attempts, 1)
        failed = result.steps[1]
        self.assertEqual(failed.action, "Move")
        self.assertTrue(failed.blocked)
        self.assertEqual(result.steps[2].position, failed.position)
        self.assertNotIn(CORRIDOR, result.route_taken)

    def test_same_observations_produce_same_decisions_despite_hidden_truth(self):
        clear = replace(SCENARIOS["clear"], sensor_overrides=((1, "clear"),))
        blocked = replace(SCENARIOS["detour"], sensor_overrides=((1, "clear"),))
        clear_result = simulate(clear, max_steps=2)
        blocked_result = simulate(blocked, max_steps=2)
        self.assertEqual(clear_result.searches, blocked_result.searches)
        for clear_step, blocked_step in zip(clear_result.steps, blocked_result.steps):
            self.assertEqual(replace(clear_step, blocked=False), replace(blocked_step, blocked=False))
        self.assertFalse(clear_result.steps[-1].blocked)
        self.assertTrue(blocked_result.steps[-1].blocked)

    def test_persistent_single_passage_stops_at_step_limit(self):
        closed = replace(SCENARIOS["temporary"], blocked_until=None)
        result = simulate(closed, max_steps=5)
        self.assertEqual(result.status, "step_limit")
        self.assertEqual(len(result.steps), 5)
        self.assertEqual(result.route_taken[-1], (2, 1))
        self.assertEqual(result.blocked_attempts, 0)

    def test_delivery_on_last_allowed_step_is_success(self):
        self.assertEqual(simulate(SCENARIOS["clear"], max_steps=4).status, "delivered")
        self.assertEqual(simulate(SCENARIOS["clear"], max_steps=0).status, "step_limit")
        with self.assertRaises(ValueError):
            simulate(SCENARIOS["clear"], max_steps=-1)

    def test_routes_and_metrics_are_consistent(self):
        for name, scenario in SCENARIOS.items():
            with self.subTest(scenario=name):
                result = simulate(scenario)
                self.assertEqual(len(result.steps), result.moves + result.waits + result.blocked_attempts)
                self.assertFalse(set(result.route_taken) & scenario.walls)
                for before, after in zip(result.route_taken, result.route_taken[1:]):
                    self.assertEqual(abs(before[0] - after[0]) + abs(before[1] - after[1]), 1)
                self.assertLessEqual(len(result.steps), 100)

    def test_console_trace_explains_all_three_algorithms(self):
        output = StringIO()
        simulate(SCENARIOS["detour"], output=output)
        text = output.getvalue()
        for explanation in ["A* input", "Expanded cell", "Bayes input", "P(blocked|signal)", "MDP input", "Q(Move)", "Q(Wait)", "MDP output: Wait", "Result: delivered"]:
            self.assertIn(explanation, text)


class CLITests(unittest.TestCase):
    def test_all_scenarios_run_from_entry_point(self):
        root = Path(__file__).resolve().parents[1]
        run = subprocess.run(
            [sys.executable, "-B", str(root / "main.py"), "--scenario", "all"],
            cwd=root, capture_output=True, text=True, check=True, timeout=10,
        )
        for name in SCENARIOS:
            self.assertIn(f"=== {name}:", run.stdout)
        self.assertIn("Delivery evaluation", run.stdout)
        self.assertEqual(run.stderr, "")


if __name__ == "__main__":
    unittest.main()
