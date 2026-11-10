"""Select a bounded set of replacements against a fixed rendered baseline."""


def select_replacements(evaluate, candidate_ids, *, max_trials):
    """Return shortlist indices and the evidence for the exact committed set.

    The callback evaluates shortlist indices together, without changing the live
    scene. After a failed batch, try the halves in shortlist order and verify
    each addition with the accepted prefix. This preserves the ordering used
    to render the proposals and accounts for interactions between candidates.
    """
    candidate_ids = tuple(candidate_ids)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('Replacement candidate IDs must be unique.')
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError('The replacement trial budget must be a positive integer.')

    def ids(indices):
        return [candidate_ids[i] for i in indices]

    trials = []
    rejected = []
    pending = []

    def probe(prefix, additions, phase):
        result = evaluate(prefix + additions)
        if type(result.get('accepted')) is not bool:
            raise ValueError('The replacement evaluator must return a boolean accepted result.')
        trial = dict(index=len(trials), phase=phase, candidate_ids=ids(prefix + additions),
                     added_candidate_ids=ids(additions), accepted_prefix_ids=ids(prefix), **result)
        trials.append(trial)
        return trial

    def split(group):
        middle = len(group) // 2
        pending.extend((group[middle:], group[:middle]))

    chosen = ()
    final_trial = None
    all_indices = tuple(range(len(candidate_ids)))
    if all_indices:
        initial = probe((), all_indices, 'initial')
        if initial['accepted']:
            chosen, final_trial = all_indices, initial
        elif len(all_indices) == 1:
            rejected.append(dict(candidate_id=candidate_ids[0], trial_index=initial['index']))
        else:
            split(all_indices)
            # Keep one trial available to confirm the recovered combination.
            while pending and len(trials) < max_trials - 1:
                group = pending.pop()
                trial = probe(chosen, group, 'addition')
                if trial['accepted']:
                    chosen += group
                elif len(group) > 1:
                    split(group)
                else:
                    rejected.append(dict(candidate_id=candidate_ids[group[0]], trial_index=trial['index']))
            if chosen:
                final_trial = probe((), chosen, 'final')
                if not final_trial['accepted']:
                    chosen = ()

    return chosen, dict(proposed=len(candidate_ids), accepted=bool(chosen),
                        selected_candidate_ids=ids(chosen), trial_count=len(trials), max_trials=max_trials,
                        trials=trials, final_trial=final_trial, rejected_candidates=rejected,
                        unresolved_candidate_ids=ids(sorted(i for group in pending for i in group)),
                        budget_exhausted=bool(pending))
