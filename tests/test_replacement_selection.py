import unittest

from shuolee_portrait.replacement_selection import select_replacements


class ReplacementSelectionTests(unittest.TestCase):
    def test_rejected_batch_recovers_audited_safe_subset(self):
        calls = []

        def evaluate(indices):
            calls.append(indices)
            failed = 1 in indices or (0 in indices and 3 in indices)
            return dict(accepted=not failed, failures=[
                dict(frame='frame_000041', checks=['foreground_holes'])
            ] if failed else [])

        chosen, audit = select_replacements(evaluate, [60, 4, 30, 8], max_trials=8)

        self.assertEqual(chosen, (0, 2))
        self.assertEqual(calls[0], (0, 1, 2, 3))
        self.assertEqual(calls[-1], chosen)
        self.assertEqual(audit['selected_candidate_ids'], [60, 30])
        self.assertEqual(audit['final_trial']['phase'], 'final')
        self.assertEqual(audit['final_trial']['candidate_ids'], [60, 30])
        self.assertEqual(audit['trial_count'], len(calls))
        rejections = audit['rejected_candidates']
        self.assertEqual([item['candidate_id'] for item in rejections], [4, 8])
        interacting = audit['trials'][rejections[-1]['trial_index']]
        self.assertEqual(interacting['accepted_prefix_ids'], [60, 30])
        self.assertEqual(interacting['added_candidate_ids'], [8])
        self.assertEqual(interacting['failures'], [
            dict(frame='frame_000041', checks=['foreground_holes'])
        ])

    def test_budget_leaves_unresolved_candidates_and_confirms_the_exact_set(self):
        def evaluate(indices):
            return dict(accepted=len(indices) <= 2)

        chosen, audit = select_replacements(evaluate, range(4), max_trials=3)

        self.assertEqual(chosen, (0, 1))
        self.assertEqual(audit['trial_count'], 3)
        self.assertTrue(audit['budget_exhausted'])
        self.assertEqual(audit['unresolved_candidate_ids'], [2, 3])
        self.assertEqual(audit['rejected_candidates'], [])

    def test_final_failure_keeps_original_scene(self):
        outcomes = iter((False, True, False, False))
        chosen, audit = select_replacements(
            lambda indices: dict(accepted=next(outcomes)), [2, 5], max_trials=4)

        self.assertEqual(chosen, ())
        self.assertFalse(audit['accepted'])
        self.assertEqual(audit['final_trial']['candidate_ids'], [2])
        self.assertFalse(audit['final_trial']['accepted'])

    def test_rejected_sixteen_resolves_every_candidate_within_budget(self):
        chosen, audit = select_replacements(
            lambda indices: dict(accepted=False), range(16), max_trials=32)

        self.assertEqual(chosen, ())
        self.assertEqual(audit['trial_count'], 31)
        self.assertFalse(audit['budget_exhausted'])
        self.assertEqual([row['candidate_id'] for row in audit['rejected_candidates']], list(range(16)))

    def test_empty_shortlist_performs_no_trial(self):
        chosen, audit = select_replacements(lambda indices: self.fail('Unexpected trial'), [], max_trials=32)

        self.assertEqual(chosen, ())
        self.assertEqual(audit['trial_count'], 0)
        self.assertIsNone(audit['final_trial'])
