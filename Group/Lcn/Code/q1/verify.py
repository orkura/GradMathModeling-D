"""Independent ray-box geometry, physical replay and integer-program audit.

No solver geometry/evaluate/DP functions are imported. MILP receives a separately
enumerated pattern set; HiGHS optimality is a floating-point numerical check.
"""
from collections import Counter
from itertools import product
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def independent_geometry(data, terrain, node):
    depot, dest = data['nodes']['O01'], data['nodes'][node]
    start = terrain.xy(depot['lon'], depot['lat'])
    end = terrain.xy(dest['lon'], dest['lat'])
    # Enumerate the complete ray bounding rectangle, then use slab intersection.
    inverse = ~terrain.transform
    a = np.asarray(inverse @ (depot['lon'], depot['lat']))
    b = np.asarray(inverse @ (dest['lon'], dest['lat']))
    lo = np.maximum(np.floor(np.minimum(a, b)).astype(int)-1, 0)
    hi = np.minimum(np.floor(np.maximum(a, b)).astype(int)+1, [terrain.width-1, terrain.height-1])
    cc, rr = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
    lower = np.column_stack((cc.ravel(), rr.ravel()))
    enter, leave = np.zeros(len(lower)), np.ones(len(lower))
    for axis in (0, 1):
        delta = b[axis]-a[axis]
        if abs(delta) < 1e-12:
            leave[(a[axis] < lower[:, axis]-1e-9) | (a[axis] > lower[:, axis]+1+1e-9)] = -1
        else:
            t0 = (lower[:, axis]-a[axis])/delta
            t1 = (lower[:, axis]+1-a[axis])/delta
            enter = np.maximum(enter, np.minimum(t0, t1))
            leave = np.minimum(leave, np.maximum(t0, t1))
    hit = lower[enter <= leave+1e-10]
    z = terrain.dem[hit[:, 1], hit[:, 0]]
    if not np.all(np.isfinite(z)) or (terrain.nodata is not None and np.any(z==terrain.nodata)):
        raise ValueError('Invalid terrain cells')
    height = max(float(z.max())+50, depot['z'], dest['z']+30)
    return math.dist(start, end), height-depot['z'], height-dest['z']-30, height, float(z.max())


def replay(spec, geo, mass, count):
    distance, up, back, _, _ = geo
    energy = horizontal = climb = 0.0
    for load, ascent in ((mass, up), (0.0, back)):
        equivalent = spec['range_empty'] + (spec['range_full']-spec['range_empty'])*(load/spec['payload'])**1.5
        horizontal += spec['energy']*distance/equivalent
        climb += (spec['mass_empty']+load)*9.80665*ascent/spec['eta_up']/3600000
    energy = horizontal+climb
    time = spec['prepare']+count*spec['load_per_box']+spec['service_base']+count*spec['service_per_box']
    for ascent, descent in ((up, back), (back, up)):
        time += ascent/spec['up_speed']+distance/spec['speed']+descent/spec['down_speed']
    return energy, time, 1-energy/spec['energy'], horizontal, climb


def milp_check(demand, patterns, order, expected):
    matrix = np.asarray([p['counts'] for p in patterns], dtype=float).T
    costs = np.asarray([[1, p['energy_kwh'], p['duration_s']] for p in patterns]).T
    constraints = [LinearConstraint(matrix, demand, demand)]
    records = []
    for j in order:
        res = milp(costs[j], integrality=np.ones(len(patterns)),
                   bounds=Bounds(0, np.inf), constraints=constraints,
                   options={'time_limit': 30, 'mip_rel_gap': 0})
        if res.status != 0:
            raise AssertionError(f'MILP did not establish optimality: {res.message}')
        counts = np.rint(res.x)
        if not np.allclose(matrix@counts, demand, rtol=0, atol=1e-7):
            raise AssertionError('Nonintegral or inconsistent MILP incumbent')
        value = float(costs[j]@counts)
        tolerance = (1e-6, 2e-6, 2e-4)[j]
        if abs(value-expected[j]) > tolerance:
            raise AssertionError(f'DP/MILP mismatch: component {j}, {value}, {expected[j]}')
        records.append(dict(component=j, optimum=value, dp_value=expected[j],
                            gap=float(res.mip_gap), bound=float(res.mip_dual_bound), status='OPTIMAL'))
        epsilon = (0, 1e-8, 1e-7)[j]
        constraints.append(LinearConstraint(costs[j], value-epsilon, value+epsilon))
    return records


def verify(result, data, terrain, optimality=True):
    errors, checks = [], 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    boxes = {b['id']: b for b in data['boxes']}
    seen = Counter(b for t in result['trips'] for b in t['box_ids'])
    check(seen == Counter(boxes.keys()), 'Each original box must occur exactly once')
    check(len({t['id'] for t in result['trips']}) == len(result['trips']), 'Unique trip IDs')
    energy, duration, socs, horizontal, climb = 0.0, 0.0, [], 0.0, 0.0
    geometries = {n: independent_geometry(data, terrain, n) for n in sorted(set(b['node'] for b in boxes.values()))}
    for t in result['trips']:
        bs = [boxes[b] for b in t['box_ids'] if b in boxes]
        check(bool(bs) and all(b['node']==t['node'] for b in bs), f"Destination {t['id']}")
        mass, volume = sum(b['mass'] for b in bs), sum(b['volume'] for b in bs)
        spec = data['types'][t['type']]
        check(mass <= spec['payload']+1e-9 and volume <= spec['volume']+1e-10, f"Capacity {t['id']}")
        vals = replay(spec, geometries[t['node']], mass, len(bs))
        for key, v in zip(('energy_kwh', 'duration_s', 'return_soc', 'horizontal_kwh', 'climb_kwh'), vals):
            check(math.isclose(t[key], v, rel_tol=0, abs_tol=1e-6), f"Physical replay {key} {t['id']}")
        check(abs(t['mass_kg']-mass)<1e-8 and abs(t['volume_m3']-volume)<1e-9, f"Cargo totals {t['id']}")
        check(vals[2]>=result['reserve']-1e-10, f"Reserve {t['id']}")
        energy += vals[0]
        duration += vals[1]
        socs.append(vals[2])
        horizontal += vals[3]
        climb += vals[4]
    actual = dict(energy_kwh=energy, cumulative_time_s=duration, min_return_soc=min(socs),
                  sorties=len(result['trips']), boxes=sum(seen.values()),
                  mass_kg=sum(boxes[b]['mass']*n for b,n in seen.items()),
                  volume_m3=sum(boxes[b]['volume']*n for b,n in seen.items()),
                  horizontal_kwh=horizontal, climb_kwh=climb)
    for k, v in actual.items():
        check(abs(result['metrics'][k]-v)<1e-6, f'Aggregate {k}')
    check(result['metrics']['sorties_by_type']==dict((k, sum(t['type']==k for t in result['trips'])) for k in 'ABC'), 'Type counts')
    certificates = []
    for area in result['areas']:
        node = area['geometry']['node']
        d, up, back, h, z = geometries[node]
        for key, value in zip(('distance_m','out_up_m','back_up_m','cruise_m','dem_max_m'), (d,up,back,h,z)):
            check(abs(area['geometry'][key]-value)<1e-5, f'Geometry {node} {key}')
        groups = {}
        for b in boxes.values():
            if b['node']==node:
                groups.setdefault((b['kind'], b['mass'], b['volume']), []).append(b['id'])
        keys = sorted(groups)
        demand = [len(groups[k]) for k in keys]
        check(area['demand']==demand, f'Demand {node}')
        rebuilt = []
        for vector in product(*(range(n+1) for n in demand)):
            if not any(vector):
                continue
            m = sum(v*k[1] for v,k in zip(vector,keys))
            vol = sum(v*k[2] for v,k in zip(vector,keys))
            for tid, spec in sorted(data['types'].items()):
                if m > spec['payload']+1e-9 or vol > spec['volume']+1e-10:
                    continue
                e,t,soc,_,_ = replay(spec, geometries[node], m, sum(vector))
                if soc >= result['reserve']-1e-12:
                    rebuilt.append(dict(type=tid, counts=list(vector), energy_kwh=e, duration_s=t))
        expected_patterns = {(p['type'],tuple(p['counts'])) for p in rebuilt}
        check(expected_patterns=={(p['type'],tuple(p['counts'])) for p in area['patterns']}, f'Pattern completeness {node}')
        local = [t for t in result['trips'] if t['node']==node]
        totals = [len(local), sum(t['energy_kwh'] for t in local), sum(t['duration_s'] for t in local)]
        check(all(abs(a-b)<1e-6 for a,b in zip(totals, area['objective'])), f'Local objective {node}')
        if optimality:
            records = milp_check(demand, rebuilt, result['objective_order'], totals)
            certificates.append(dict(node=node, phases=records))
            checks += len(records)
    for c in result['safe_payloads']:
        q, spec = c['safe_payload_kg'], data['types'][c['type']]
        geo = geometries[c['node']]
        if q is None:
            check(replay(spec, geo, 0, 0)[2] < result['reserve'], 'Unreachable capacity')
        else:
            check(0 <= q <= spec['payload'] and replay(spec,geo,q,0)[2]>=result['reserve']-1e-9, 'Safe capacity')
            if q < spec['payload']-1e-5:
                check(replay(spec,geo,q+1e-5,0)[2]<result['reserve'], 'Maximal capacity')
    return dict(passed=not errors, checks=checks, errors=errors, milp_certificates=certificates,
                scope='Exhaustive finite pattern model; float64 physics, numerical MILP optimality',
                tolerance=dict(energy_kwh=2e-6, time_s=2e-4, geometry_m=1e-5))
