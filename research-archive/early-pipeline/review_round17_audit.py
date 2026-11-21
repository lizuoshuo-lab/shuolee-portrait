"""Verify singleton optimized children and separately account for the R15 fallback."""

from math import isfinite, isclose
from pathlib import Path

import numpy as np
from PIL import Image

import review_smpl_delivery_round16 as prior


read, pin, require = prior.read, prior.pin, prior.require


def native_check(check, frames, count, *, radius=False):
    require(check['proposed'] == count and [row['frame'] for row in check['frames']] == frames,
            'Size decision was not checked on the exact ordered 24-view inventory.')
    if check['accepted']:
        require(all(not row['failures'] for row in check['frames']), 'An accepted size decision failed a native guard.')
        if radius:
            evidence = check['native_radius_check']
            parent, child = evidence['before_px'], evidence['child_max_px']
            require(evidence['all_decreased'] and len(parent) == len(child) == count
                    and all(a > b and a > evidence['limit_px'] for a,b in zip(parent,child,strict=True)),
                    'A committed child does not reduce its parent native raster radius.')


def verify(audit, integration, frames):
    events, totals, summary = audit['events'], audit['totals'], audit['candidate_optimization']
    require(audit['state'] == 'completed' and len(frames) == 24 and len(set(frames)) == 24
            and len(events) == audit['stock_events'] == integration['stock_densifications']
            and all(a['step'] < b['step'] for a,b in zip(events,events[1:]))
            and audit['terminal'] == integration['densification_audit']['terminal']
            and audit['terminal']['live_rows_over_local_cap'] == 0
            and summary == integration['densification_audit']['candidate_optimization']
            and summary['main_optimizer_updates'] == integration['optimizer_updates'] == 10000,
            'The R17 audit and terminal configuration differ.')
    require(all(n == sum(event.get(key,0) for event in events) for key,n in totals.items()),
            'Size decision totals differ from event records.')
    attempts, evaluated, retired, born = [],set(),set(),set()
    for event in events:
        before, after = (event[f'screen_candidates_{position}_stock'] for position in ('before','after'))
        require(0 <= after <= before and before-after == event['screen_parents_replaced_by_gradient_split'],
                'The screen candidate topology accounting differs.')
        if event['screen_limit_px'] is None:
            require(before == after == 0 and not event.get('local_optimizations'), 'Unexpected early size optimization.')
            continue
        candidates, removed, replaced = (event[key] for key in ('size_candidates','size_pruned','size_replaced'))
        require(after == event['screen_candidates'] <= candidates
                and 0 <= event['foreground_supported'] <= event['assessed'] <= candidates
                and candidates == event['protected']+removed+replaced and event['screen_pruned'] <= removed,
                'Native size decision accounting differs.')
        if removed:
            trial = event['removal_trial']
            require(trial['accepted'],'Recorded size deletion lacked acceptance.')
            native_check(trial,frames,removed)
        cycle = event.get('local_optimizations', [])
        require(len(cycle) <= 8,'An event exceeded its singleton shortlist.')
        committed = []
        child_count = 0
        for attempt in cycle:
            parent, = attempt['stable_parent_ids']
            require(type(parent) is int and parent >= 0 and parent not in (evaluated|retired)
                    and attempt['prior_optimized_commits'] == committed
                    and attempt['scene_scope'] == 'singleton_replacement_in_current_live_scene',
                    'A candidate reused an ID or a different live scene.')
            evaluated.add(parent)
            accepted = attempt['committed_parent_ids']
            rejected = attempt['rejected_parent_ids']
            require(accepted+rejected == [parent] and all(isfinite(attempt[key]) and attempt[key] >= 0
                    for key in ('seconds','verification_seconds')) and type(attempt['optimizer_updates']) is int
                    and attempt['optimizer_updates'] >= 0,
                    'Invalid candidate decision or time budget.')
            views = attempt['views']
            require(len(views) == len(frames) == attempt['optimizer_view_budget']
                    and len(views) == len({view['view_index'] for view in views})
                    and all(0 <= view['view_index'] < len(frames) and view['frame'] == frames[view['view_index']]
                            for view in views), 'Local view identity differs.')
            for view in views:
                require(isfinite(view['foreground_equivalent_pixels']) and view['foreground_equivalent_pixels'] >= 0
                        and all(type(view[key]) is bool for key in ('foreground_supervision','exterior_supervision','boundary_supervision')),
                        'Candidate view support is missing or invalid.')
                for stage in ('parent','proposal','optimized'):
                    evidence = view[stage]
                    shape = (1,) if stage=='parent' else (1,attempt['children_per_parent'])
                    require(np.asarray(evidence['radii_px']).shape==shape and all(value>=0 for value in np.asarray(evidence['radii_px']).flat),
                            'Candidate native radii have the wrong lineage.')
            cycles = attempt['cycles']
            require(len(cycles) <= attempt['budget_cycles'] <= 8
                    and attempt['optimizer_updates'] <= attempt['optimizer_update_budget']==len(frames)*attempt['budget_cycles']
                    and all(cycle['cycle']==i and cycle['updates']==i*len(frames)
                            and cycle['views_checked']==len(frames) and sorted(cycle['view_order'])==sorted(frames)
                            for i,cycle in enumerate(cycles,1)), 'Local optimizer did not use ordered full native-view cycles.')
            check = attempt['replacement_trial']
            if check is not None:
                native_check(check,frames,1,radius=True)
                require(bool(accepted) == check['accepted'], 'Local decision and full-view acceptance disagree.')
            if accepted:
                commit = attempt['commit']
                children = commit['child_ids']
                require(check is not None and attempt['status'] == 'fitted' and attempt['optimizer_updates'] > 0
                        and attempt['checkpoint_feasible'] and 1 <= attempt['checkpoint_cycle'] <= len(cycles)
                        and commit['parent_ids'] == accepted and set(commit['moment_groups']) == {'xyz','scaling','opacity','f_dc'}
                        and len(children) == len(set(children)) == commit['children'] == attempt['children_per_parent'] >= 3
                        and not (set(children) & (born|retired|{parent})), 'Invalid optimized replacement lineage or Adam state.')
                committed.append(parent)
                retired.add(parent)
                born.update(children)
                child_count += len(children)
            else:
                require('commit' not in attempt, 'A rejected candidate mutated live topology.')
            attempts.append((event,attempt))
        fallback = event.get('fallback_replacement') or {}
        fallback_parents = fallback.get('committed_parent_ids', [])
        fallback_children = fallback.get('child_ids', [])
        if fallback:
            require(fallback['path'] == 'r15_fallback'
                    and set(fallback_parents) <= set(fallback['eligible_parent_ids']) <= set(fallback['shortlist_parent_ids'])
                    and not (set(fallback['shortlist_parent_ids']) & retired)
                    and len(fallback_children) == len(set(fallback_children)) == 2*len(fallback_parents)
                    and not (set(fallback_children) & (born|retired|set(fallback_parents))),
                    'R15 fallback shortlist or parent/child identity differs.')
            if fallback_parents:
                final = fallback['trial']['final_trial']
                require(final['accepted'] and len(fallback['trial']['selected_candidate_ids']) == len(fallback_parents),
                        'R15 fallback commit did not pass its own decision.')
                native_check(final,frames,len(fallback_parents),radius=True)
            retired.update(fallback_parents)
            born.update(fallback_children)
        require(len(committed) == event['optimized_size_replaced']
                and len(fallback_parents) == event['fallback_size_replaced']
                and child_count == event['optimized_replacement_children']
                and len(fallback_children) == event['fallback_replacement_children']
                and replaced == len(committed)+len(fallback_parents)
                and event['replacement_children'] == child_count+len(fallback_children),
                'Optimized and fallback replacement counts were mixed.')
    require(summary['events'] == sum(bool(event.get('local_optimizations')) for event in events)
            and summary['attempts'] == summary['optimized_parent_events'] == summary['unique_optimized_parents'] == len(attempts)
            and summary['repeated_optimized_parent_events'] == 0
            and summary['size_candidate_events'] == totals['size_candidates']
            and summary['size_candidate_events']-summary['unique_size_candidate_parents'] == summary['repeated_size_candidate_events']
            and summary['optimized_parent_commits'] == totals['optimized_size_replaced']
            and summary['fallback_parent_commits'] == totals['fallback_size_replaced']
            and summary['optimizer_updates'] == sum(attempt['optimizer_updates'] for _,attempt in attempts)
            and all(isclose(summary[key],sum(attempt[key] for _,attempt in attempts),abs_tol=1e-8)
                    for key in ('seconds','verification_seconds')),
            'Optimized/fallback totals, updates or elapsed times differ.')
    return summary,attempts


def event_proofs(run,attempts):
    events = {event['step']:event for event,_ in attempts}
    proof = {}
    for step,event in events.items():
        path = run/'gaussian-run/size-child-diagnostics'/f'step-{step:06d}'/'event.json'
        require(read(path)==event,'Live size decision event differs from the terminal audit.')
        proof[f'candidate_event_{step:06d}'] = prior.descriptor(path)
    return proof


def panels(run, attempts):
    """Read back selected real parent→proposal→optimized→actual decision panels."""
    paired = [(event,attempt) for event,attempt in attempts
              if attempt.get('representatives') and attempt.get('decision_representatives')]
    selected = []
    for committed in (True,False):
        first = next(((event,attempt) for event,attempt in paired if bool(attempt['committed_parent_ids'])==committed),None)
        if first:
            selected.append(first)
    rows = {row['frame']:row for row in read(run/'data/training-inputs.json')['records']}
    result = []
    for event,attempt in selected:
        step,parent = event['step'],attempt['stable_parent_ids'][0]
        folder = f'step-{step:06d}/parent-{parent:07d}'
        require(attempt['diagnostic_subdir']==folder,'Unexpected candidate diagnostic directory.')
        decisions = {item['frame']:item for item in attempt['decision_representatives']}
        metrics = {item['frame']:item for item in attempt['views']}
        require(set(decisions)=={item['frame'] for item in attempt['representatives']}, 'Candidate diagnostic pairing differs.')
        for image in attempt['representatives']:
            frame = image['frame']
            row = rows[frame]
            box = image['box']
            x0,y0,x1,y1 = box
            width,height = x1-x0,y1-y0
            stride = max(width,105)+3
            require(0<=x0<x1<=row['wh'][0] and 0<=y0<y1<=row['wh'][1], 'Invalid native trial crop.')
            with Image.open(pin(row['image'],row['source_sha256'])) as raw:
                source = np.asarray(raw.convert('RGB').crop(box))
            files, crops = [],[]
            for kind,count,item in (('representative',4,image),('decision',2,decisions[frame])):
                name = f'{kind}-view-{metrics[frame]["view_index"]:02d}.png'
                require(item['path']==name and item['box']==box,'Trial stage and live scene crop identity differ.')
                path = pin(run/'gaussian-run/size-child-diagnostics'/folder/name,item['sha256'])
                with Image.open(path) as decoded:
                    require(decoded.format=='PNG' and decoded.mode=='RGB' and decoded.size==(count*stride+3,2*height+39),
                            'Candidate panel layout changed.')
                    pixels = np.asarray(decoded)
                require(np.array_equal(source,pixels[20:20+height,3:3+width]), 'Candidate source RGB pixels differ.')
                crops.append(pixels[20:23+2*height,3+stride:3+stride+width])
                files.append(dict(prior.descriptor(path),frame=frame,step=step,parent_id=parent,
                    view=f'size_child_{kind}',native_crop=box,committed=bool(attempt['committed_parent_ids']),
                    decision_reason=attempt['decision_reason'],raw_view_metrics=metrics[frame]))
            delta = int(np.abs(crops[0].astype(np.int16)-crops[1]).max())
            result.extend(dict(item,parent_to_live_max_abs_8bit=delta) for item in files)
    return result
