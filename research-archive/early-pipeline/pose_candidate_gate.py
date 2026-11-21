"""A region guard for saved pose candidates, not anatomical acceptance.

The caller supplies measurement groups and limits. No frame or dataset identity
is used. Missing baseline evidence fails closed instead of admitting a pose.
"""
import math


def choose_candidate(scores, baseline, candidate, limits):
    if len(scores) != 2 or not math.isfinite(scores[1]):
        raise ValueError('A finite baseline objective is required')
    if not limits or any(not math.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Explicit nonnegative regional limits are required')
    reasons, deltas = [], {}
    for group, limit in limits.items():
        old = baseline.get(group)
        new = candidate.get(group)
        if not old or old.get('count', 0) <= 0 or not math.isfinite(old.get('median', math.nan)):
            raise ValueError('Baseline measurement unavailable: ' + group)
        if (not new or new.get('count') != old['count']
                or not math.isfinite(new.get('median', math.nan))):
            reasons.append(group + ': unsupported candidate measurement')
            continue
        delta = new['median'] - old['median']
        deltas[group] = delta
        if delta > limit:
            reasons.append(group + ': regression exceeds registered limit')
    if not math.isfinite(scores[0]) or scores[0] >= scores[1]:
        reasons.append('Original objective does not strictly improve')
    return dict(selected_index=1 if reasons else 0, reasons=reasons,
                regional_deltas_px=deltas, quality_accepted=False)
