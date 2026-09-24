"""Exhaustive category-count DP; no fleet, deadlines or radio constraints.

Integers represent cargo grams and cubic centimetres. Energy/time use float64;
lexicographic comparisons identify ties within 1e-9 kWh and 1e-6 seconds.
All feasible patterns and states are enumerated, with no search cutoff.
"""
from collections import defaultdict
from functools import lru_cache
from itertools import product
import math

G = 9.80665
ORDERS = {"sorties_first": (0, 1, 2), "energy_first": (1, 0, 2),
          "time_first": (2, 0, 1)}


def better(a, b, order=(0, 1, 2)):
    for j in order:
        if abs(a[j] - b[j]) > (0, 1e-9, 1e-6)[j]:
            return a[j] < b[j]
    return False


def geometry(data, terrain, node):
    a, b = data['nodes']['O01'], data['nodes'][node]
    x, y = terrain.xy(a['lon'], a['lat']), terrain.xy(b['lon'], b['lat'])
    highest = terrain.max_along(x, y)
    cruise = max(highest + 50, a['z'], b['z'] + 30)
    return dict(node=node, distance_m=float(math.dist(x, y)), dem_max_m=highest,
                cruise_m=cruise, out_up_m=cruise-a['z'], back_up_m=cruise-b['z']-30)


def evaluate(spec, geo, mass, count):
    length = spec['range_empty'] - (spec['range_empty']-spec['range_full']) * (mass/spec['payload'])**1.5
    horizontal = spec['energy'] * geo['distance_m'] * (1/length+1/spec['range_empty'])
    climb = G * ((spec['mass_empty']+mass)*geo['out_up_m'] +
                 spec['mass_empty']*geo['back_up_m']) / (3.6e6*spec['eta_up'])
    duration = (2*geo['distance_m']/spec['speed'] +
                (geo['out_up_m']+geo['back_up_m'])*(1/spec['up_speed']+1/spec['down_speed']) +
                spec['prepare']+count*spec['load_per_box'] +
                spec['service_base']+count*spec['service_per_box'])
    return dict(energy_kwh=horizontal+climb, horizontal_kwh=horizontal, climb_kwh=climb,
                duration_s=duration, return_soc=1-(horizontal+climb)/spec['energy'])


def safe_payload(spec, geo, reserve):
    cap = (1-reserve)*spec['energy']
    if evaluate(spec, geo, 0, 0)['energy_kwh'] > cap:
        return None
    if evaluate(spec, geo, spec['payload'], 0)['energy_kwh'] <= cap:
        return spec['payload']
    lo, hi = 0.0, spec['payload']
    for _ in range(70):
        mid = (lo+hi)/2
        if evaluate(spec, geo, mid, 0)['energy_kwh'] <= cap:
            lo = mid
        else:
            hi = mid
    return lo


def categories(data, node):
    groups = defaultdict(list)
    for b in data['boxes']:
        if b['node'] == node:
            groups[(b['kind'], round(b['mass']*1000), round(b['volume']*1_000_000))].append(b['id'])
    return [dict(kind=k[0], grams=k[1], cm3=k[2], ids=sorted(v)) for k, v in sorted(groups.items())]


def enumerate_patterns(data, geo, groups, reserve):
    patterns = []
    for counts in product(*(range(len(g['ids'])+1) for g in groups)):
        n = sum(counts)
        if n == 0:
            continue
        grams = sum(c*g['grams'] for c, g in zip(counts, groups))
        cm3 = sum(c*g['cm3'] for c, g in zip(counts, groups))
        for tid, spec in sorted(data['types'].items()):
            if grams > round(spec['payload']*1000) or cm3 > round(spec['volume']*1_000_000):
                continue
            values = evaluate(spec, geo, grams/1000, n)
            if values['return_soc'] < reserve-1e-12:
                continue
            patterns.append(dict(type=tid, counts=list(counts), mass_kg=grams/1000,
                                 volume_m3=cm3/1_000_000, **values))
    return patterns


def dynamic_program(demand, patterns, order):
    choices = {}

    @lru_cache(None)
    def visit(state):
        if not any(state):
            return (0, 0.0, 0.0)
        best = (math.inf, math.inf, math.inf)
        for pno, p in enumerate(patterns):
            if any(c > s for c, s in zip(p['counts'], state)):
                continue
            prev = tuple(s-c for s, c in zip(state, p['counts']))
            v = visit(prev)
            candidate = (v[0]+1, v[1]+p['energy_kwh'], v[2]+p['duration_s'])
            if better(candidate, best, order):
                best = candidate
                choices[state] = (prev, pno)
        return best

    cost = visit(tuple(demand))
    if not math.isfinite(cost[0]):
        raise ValueError(f'No feasible batching for demand {demand}')
    selected, state = [], tuple(demand)
    while any(state):
        state, pno = choices[state]
        selected.append(pno)
    return cost, selected, visit.cache_info().currsize


def summarize(trips):
    return dict(sorties=len(trips), boxes=sum(len(t['box_ids']) for t in trips),
                mass_kg=sum(t['mass_kg'] for t in trips),
                volume_m3=sum(t['volume_m3'] for t in trips),
                energy_kwh=sum(t['energy_kwh'] for t in trips),
                horizontal_kwh=sum(t['horizontal_kwh'] for t in trips),
                climb_kwh=sum(t['climb_kwh'] for t in trips),
                cumulative_time_s=sum(t['duration_s'] for t in trips),
                min_return_soc=min(t['return_soc'] for t in trips),
                sorties_by_type={tid: sum(t['type']==tid for t in trips) for tid in 'ABC'})


def solve(data, terrain, reserve=.2, priority='sorties_first'):
    trips, areas, capacities = [], [], []
    for node in sorted(set(b['node'] for b in data['boxes'])):
        geo, groups = geometry(data, terrain, node), categories(data, node)
        patterns = enumerate_patterns(data, geo, groups, reserve)
        demand = [len(g['ids']) for g in groups]
        cost, selected, states = dynamic_program(demand, patterns, ORDERS[priority])
        used = [0]*len(groups)
        for pno in selected:
            p = patterns[pno]
            ids = []
            for j, c in enumerate(p['counts']):
                ids.extend(groups[j]['ids'][used[j]:used[j]+c])
                used[j] += c
            trips.append(dict(p, id=f'Q1-{len(trips)+1:03d}', node=node, box_ids=sorted(ids)))
        areas.append(dict(geometry=geo, categories=groups, demand=demand,
                          patterns=patterns, selected=selected, objective=list(cost), states=states))
        for tid, spec in sorted(data['types'].items()):
            capacities.append(dict(node=node, type=tid, reserve=reserve,
                                   safe_payload_kg=safe_payload(spec, geo, reserve)))
    return dict(reserve=reserve, priority=priority, objective_order=list(ORDERS[priority]),
                metrics=summarize(trips), trips=trips, areas=areas, safe_payloads=capacities)
