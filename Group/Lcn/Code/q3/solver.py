"""Deterministic multi-start constructive search for the Q3 joint schedule.

Search is deliberately bounded: sampled relay candidates, at most three stops
per transport sortie, one guaranteed relay provider per transport sortie.
Every accepted trajectory chunk is checked with a conservative continuous
link bound; candidates are not advertised as a global optimum.
"""
from __future__ import annotations

import copy
import itertools
import math
import random
import time
from collections import defaultdict

import numpy as np

GRAVITY = 9.80665


def charge_time(soc, full):
    return full * (0.65 * max(0.0, 0.9 - soc) / 0.9
                   + 0.35 * min(0.1, max(0.0, 1 - soc)) / 0.1)


def flight_energy(spec, distance, up, load):
    length = spec['range_empty'] - (spec['range_empty'] - spec['range_full']) * (load / spec['payload']) ** 1.5
    return spec['energy'] * distance / length + (spec['mass_empty'] + load) * GRAVITY * up / (spec['eta_up'] * 3.6e6)


def shift_segments(segments, offset):
    return [dict(s, t0=s['t0'] + offset, t1=s['t1'] + offset) for s in segments]


class Model:
    def __init__(self, data, terrain, sites, spatial_step=100.0):
        self.data, self.terrain, self.sites = data, terrain, sites
        self.spatial_step = spatial_step
        self.boxes = {b['id']: b for b in data['boxes']}
        self.nodes = {}
        for nid, n in data['nodes'].items():
            xy = terrain.xy(n['lon'], n['lat'])
            self.nodes[nid] = np.r_[xy, n['z'] + (0 if nid == 'O01' else 30)]
        self.depot = self.nodes['O01']
        self.gateway = self.depot + [0, 0, data['links']['gateway_height']]
        self.legs = {}
        self.radio_cache = {}
        self.route_radio_cache = {}
        self.relay_geometry = {}
        self.site_coverage={s['id']:len(s.get('coverage',[])) for s in sites}
        for site in sites:
            self.relay_geometry[site['id']] = self._relay_geometry(site)

    def geometry(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        height = max(self.terrain.max_along(a[:2], b[:2]) + 50, a[2], b[2])
        return float(np.linalg.norm(a[:2] - b[:2])), float(height - a[2]), float(height - b[2]), float(height)

    def leg(self, a, b):
        key = (a, b)
        if key not in self.legs:
            self.legs[key] = self.geometry(self.nodes[a], self.nodes[b])
        return self.legs[key]

    @staticmethod
    def motion(a, b, height, spec, start, load=0):
        a, b = np.asarray(a), np.asarray(b)
        points = [a, np.r_[a[:2], height], np.r_[b[:2], height], b]
        speeds = [spec['up_speed'], spec['speed'], spec['down_speed']]
        out, now = [], start
        for p0, p1, speed, phase in zip(points[:-1], points[1:], speeds, ['climb', 'cruise', 'descent']):
            duration = float(np.linalg.norm(p1 - p0)) / speed
            if duration > 1e-10:
                out.append(dict(t0=now, t1=now + duration, p0=p0.tolist(), p1=p1.tolist(), phase=phase, load=load))
                now += duration
        return out, now

    def plan(self, type_id, box_ids, route):
        spec = self.data['types'][type_id]
        boxes = [self.boxes[x] for x in box_ids]
        mass, volume = sum(b['mass'] for b in boxes), sum(b['volume'] for b in boxes)
        if mass > spec['payload'] + 1e-9 or volume > spec['volume'] + 1e-10:
            return None
        destinations = {b['node'] for b in boxes}
        route = tuple(n for n in route if n in destinations)
        if set(route) != destinations or len(set(route)) != len(route):
            return None
        prepare = spec['prepare'] + len(box_ids) * spec['load_per_box']
        now, load, energy = prepare, mass, 0.0
        segments, deliveries = [], {}
        current = 'O01'
        for dest in (*route, 'O01'):
            distance, up, down, height = self.leg(current, dest)
            energy += flight_energy(spec, distance, up, load)
            legs, now = self.motion(self.nodes[current], self.nodes[dest], height, spec, now, load)
            segments.extend(legs)
            if dest != 'O01':
                delivered = [b for b in boxes if b['node'] == dest]
                duration = spec['service_base'] + len(delivered) * spec['service_per_box']
                position = self.nodes[dest].tolist()
                segments.append(dict(t0=now, t1=now + duration, p0=position, p1=position, phase='delivery', load=load))
                now += duration
                for box in delivered:
                    deliveries[box['id']] = now
                    load -= box['mass']
            current = dest
        if energy > spec['energy'] * (1 - spec['reserve']) + 1e-10:
            return None
        return dict(type=type_id, box_ids=list(box_ids), route=list(route), takeoff=prepare,
                    duration=now, energy=energy, soc=1-energy/spec['energy'],
                    mass=mass, volume=volume, segments=segments, deliveries=deliveries)

    def _relay_geometry(self, site):
        spec = self.data['relay']
        point = np.asarray(site['position'])
        distance, up, down, height = self.geometry(self.depot, point)
        outward, arrival = self.motion(self.depot, point, height, spec, spec['prepare'])
        backward, return_duration = self.motion(point, self.depot, height, spec, 0)
        flight = spec['power_cruise'] * (2 * distance / spec['speed']) / 3600
        flight += spec['mass'] * GRAVITY * (up + down) / (spec['eta_up'] * 3.6e6)
        return dict(outward=outward, backward=backward, arrival=arrival,
                    ready=arrival + spec['link_time'], return_duration=return_duration,
                    flight_energy=flight)

    def _radio_geometry(self, p0, p1):
        key = tuple(np.round([*p0, *p1], 6))
        if key in self.radio_cache:
            return self.radio_cache[key]
        a, b = np.asarray(p0), np.asarray(p1)
        count = max(1, int(math.ceil(float(np.linalg.norm(b-a)) / self.spatial_step)))
        pieces = []
        for j in range(count):
            f0, f1 = j/count, (j+1)/count
            x, y = a + f0*(b-a), a + f1*(b-a)
            dm = self.terrain.segment_link_lower_bound(x, y, self.gateway, 'direct')
            support = {}
            if dm < -1e-9:
                for site in self.sites:
                    # Cheap necessary endpoint filter before swept-cell checks.
                    pos = site['position']
                    if not self.terrain.link(x, pos, 'access') or not self.terrain.link(y, pos, 'access'):
                        continue
                    margin = self.terrain.segment_link_lower_bound(x, y, pos, 'access')
                    if margin >= 0:
                        support[site['id']] = margin
            pieces.append(dict(f0=f0, f1=f1, direct=dm >= -1e-9, margin=dm, support=support))
        self.radio_cache[key] = pieces
        return pieces

    def radio(self, plan):
        # Geometry does not depend on load or the number of boxes at a stop.
        key = tuple(plan['route'])
        if key not in self.route_radio_cache:
            common = {s['id'] for s in self.sites}
            needs = False
            for segment in sorted(plan['segments'], key=lambda s: s['phase'] != 'delivery'):
                for piece in self._radio_geometry(segment['p0'], segment['p1']):
                    if not piece['direct']:
                        needs = True
                        common &= set(piece['support'])
                        if not common:
                            self.route_radio_cache[key] = (True, [])
                            return True, [], []
            self.route_radio_cache[key] = (needs, sorted(common))
        needs, common = self.route_radio_cache[key]
        if needs and not common:
            return needs, common, []
        pieces = []
        for segment in plan['segments']:
            for p in self._radio_geometry(segment['p0'], segment['p1']):
                duration = segment['t1'] - segment['t0']
                pieces.append(dict(p, t0=segment['t0'] + p['f0']*duration,
                                   t1=segment['t0'] + p['f1']*duration, phase=segment['phase']))
        return needs, common, pieces

    def relay_trip(self, site_id, drone, component, start, end, identifier):
        spec, geo = self.data['relay'], self.relay_geometry[site_id]
        site = next(x for x in self.sites if x['id'] == site_id)
        arrival, ready = start + geo['arrival'], start + geo['ready']
        end = max(end, ready)
        energy = geo['flight_energy'] + (spec['power_hover'] + spec['power_comm']) * (end-arrival)/3600
        if energy > spec['energy'] * (1-spec['reserve']) + 1e-9:
            return None
        pos = site['position']
        segments = shift_segments(geo['outward'], start)
        segments.append(dict(t0=arrival,t1=end,p0=pos,p1=pos,phase='hover',load=0))
        segments += shift_segments(geo['backward'], end)
        return dict(id=identifier,site_id=site_id,drone=drone,component=component,start=start,
                    takeoff=start+spec['prepare'],ready=ready,service_end=end,
                    position=pos,return_=end+geo['return_duration'],energy=energy,
                    soc=1-energy/spec['energy'],segments=segments)


def initial_state(data):
    return dict(trips=[], relay_trips=[], drone_free={x:0.0 for t in data['types'].values() for x in t['drones']},
                battery_free={x:0.0 for t in data['types'].values() for x in t['battery_ids']})


def relay_availability(state, data):
    drones = {x:0.0 for x in data['relay']['drones']}
    components = {x:0.0 for x in data['relay']['component_ids']}
    for r in state['relay_trips']:
        drones[r['drone']] = max(drones[r['drone']], r['return'] + data['relay']['turnaround'])
        components[r['component']] = max(components[r['component']], r['return'] + charge_time(r['soc'],data['relay']['charge_full']))
    return drones, components


def place(model, state, plan, settings):
    data, spec = model.data, model.data['types'][plan['type']]
    drone = min(spec['drones'],key=lambda x:(state['drone_free'][x],x))
    battery = min(spec['battery_ids'],key=lambda x:(state['battery_free'][x],x))
    earliest = max(state['drone_free'][drone],state['battery_free'][battery])
    needs, sites, pieces = model.radio(plan)
    if needs and not sites:
        return None
    options = []
    if not needs:
        options.append((earliest, None, None, 0.0))
    else:
        required = [p for p in pieces if not p['direct']]
        first, last = min(p['t0'] for p in required), max(p['t1'] for p in required)
        drone_free, component_free = relay_availability(state,data)
        for site in sites:
            geo = model.relay_geometry[site]
            for idx, old in enumerate(state['relay_trips']):
                if old['site_id'] != site:
                    continue
                start = max(earliest,old['ready'] - first)
                end = max(old['service_end'],start+last)
                # Extending only a terminal assignment cannot invalidate a later one.
                later = any(r['id'] != old['id'] and (r['drone']==old['drone'] or r['component']==old['component'])
                            and r['start'] >= old['return']-1e-7 for r in state['relay_trips'])
                if later and end > old['service_end']+1e-7:
                    continue
                new = model.relay_trip(site,old['drone'],old['component'],old['start'],end,old['id'])
                if new is not None:
                    new['return'] = new.pop('return_')
                    options.append((start,new,idx,new['energy']-old['energy']))
            rd = min(drone_free,key=lambda x:(drone_free[x],x))
            rc = min(component_free,key=lambda x:(component_free[x],x))
            relay_start = max(drone_free[rd],component_free[rc])
            start = max(earliest,relay_start+geo['ready']-first)
            # Just-in-time launch avoids unnecessary idle hover at new sites.
            relay_start = max(relay_start,start+first-geo['ready'])
            new = model.relay_trip(site,rd,rc,relay_start,start+last,f"R{len(state['relay_trips'])+1:03d}")
            if new is not None:
                new['return'] = new.pop('return_')
                options.append((start,new,None,new['energy']))
    best = None
    for start, relay, replace, added_energy in options:
        violation = sum(max(0.0,start+t-model.boxes[b]['deadline']) for b,t in plan['deliveries'].items())
        if violation > 1e-6:
            continue
        tardiness = sum(model.boxes[b]['weight']*max(0.0,start+t-model.boxes[b]['due']) for b,t in plan['deliveries'].items())
        finish = start+plan['duration']
        newrelay = relay is not None and replace is None
        urgent=sum(math.isfinite(model.boxes[b]['deadline']) for b in plan['box_ids'])
        count=len(plan['box_ids'])+settings.get('hard_bonus',0)*urgent
        coverage=model.site_coverage.get(relay['site_id'],0) if relay else 0
        score = (finish + settings['wait_weight']*start + settings['energy_weight']*(plan['energy']+added_energy)
                 + settings['relay_weight']*newrelay + 2*tardiness
                 - settings.get('coverage_weight',0)*coverage*newrelay) / count**settings['packing_power']
        if best is None or score < best['score']:
            best = dict(score=score,start=start,drone=drone,battery=battery,relay=relay,
                        replace=replace,plan=plan,pieces=pieces)
    return best


def commit(model,state,placement):
    p, start = placement['plan'],placement['start']
    trip = {k:v for k,v in p.items() if k not in ('duration','segments','deliveries')}
    trip.update(id=f"T{len(state['trips'])+1:03d}",drone=placement['drone'],battery=placement['battery'],
                start=start,takeoff=start+p['takeoff'],segments=shift_segments(p['segments'],start),
                deliveries={b:t+start for b,t in p['deliveries'].items()})
    trip['return'] = start+p['duration']
    relay = placement['relay']
    trip['relay_trip_id'] = relay['id'] if relay else None
    if relay:
        if placement['replace'] is None:
            state['relay_trips'].append(relay)
        else:
            state['relay_trips'][placement['replace']] = relay
    state['trips'].append(trip)
    state['drone_free'][trip['drone']] = trip['return']
    state['battery_free'][trip['battery']] = trip['return']+charge_time(trip['soc'],model.data['types'][trip['type']]['charge_full'])
    return trip


def candidate_plans(model,remaining,seed,settings,rng):
    bynode=defaultdict(list)
    for b in remaining:
        bynode[model.boxes[b]['node']].append(b)
    seednode=model.boxes[seed]['node']
    nearby=sorted((n for n in bynode if n!=seednode),key=lambda n:np.linalg.norm(model.nodes[n][:2]-model.nodes[seednode][:2]))[:settings['neighbors']]
    routes=[(seednode,)]
    if settings['max_stops']>=2:
        for n in nearby:
            routes.extend([(seednode,n),(n,seednode)])
    if settings['max_stops']>=3:
        for a,b in itertools.combinations(nearby[:3],2):
            routes.extend([(seednode,a,b),(a,seednode,b),(a,b,seednode)])
    hard_remains=any(math.isfinite(model.boxes[b]['deadline']) for b in remaining)
    noise={b:rng.random() for b in remaining}
    unique={}
    for route in routes:
        pool=[b for n in route for b in bynode[n] if b!=seed]
        pool.sort(key=lambda b:(model.boxes[b]['deadline'],model.boxes[b]['due'],
                                -model.boxes[b]['weight']+settings['noise']*noise[b]))
        if settings['hard_first'] and hard_remains:
            pool=[b for b in pool if math.isfinite(model.boxes[b]['deadline'])]
        for tid in model.data['types']:
            batch=[seed]
            best=model.plan(tid,batch,route)
            if best is None:
                continue
            # Retain an emergency seed-only alternative; a fully packed sortie
            # can miss a deadline even though a smaller dispatch is feasible.
            unique[(tid,tuple(batch),tuple(best['route']))]=best
            for b in pool:
                test=model.plan(tid,batch+[b],route)
                if test is not None:
                    batch.append(b)
                    best=test
                    if all(math.isfinite(model.boxes[x]['deadline']) for x in batch):
                        unique[(tid,tuple(sorted(batch)),tuple(best['route']))]=best
            unique[(tid,tuple(sorted(batch)),tuple(best['route']))]=best
    # Evaluate a diverse short list, retaining every single-stop alternative.
    ranked=sorted(unique.values(),key=lambda p:(p['duration']+settings['energy_weight']*p['energy'])/len(p['box_ids'])**settings['packing_power'])
    retained=ranked[:settings['plan_limit']]
    retained += [p for p in ranked[settings['plan_limit']:] if len(p['route'])==1]
    return retained


def metrics(result,model):
    trips,relays=result['trips'],result['relay_trips']
    delivery={b:t for p in trips for b,t in p['deliveries'].items()}
    te=sum(p['energy'] for p in trips)
    re=sum(p['energy'] for p in relays)
    return dict(boxes_delivered=len(delivery),transport_sorties=len(trips),relay_sorties=len(relays),
                transport_energy_kwh=te,relay_energy_kwh=re,total_energy_kwh=te+re,
                makespan_s=max([p['return'] for p in trips+relays],default=0),
                transport_makespan_s=max([p['return'] for p in trips],default=0),
                weighted_tardiness=sum(model.boxes[b]['weight']*max(0,t-model.boxes[b]['due']) for b,t in delivery.items()),
                hard_deadline_violations=sum(t>model.boxes[b]['deadline']+1e-6 for b,t in delivery.items()),
                min_transport_soc=min([p['soc'] for p in trips],default=1),min_relay_soc=min([p['soc'] for p in relays],default=1))


def construct(model,settings,seed):
    rng=random.Random(seed)
    remaining=set(model.boxes)
    state=initial_state(model.data)
    ranks={b:rng.random() for b in sorted(remaining)}
    nominal={}
    for b in remaining:
        box=model.boxes[b]
        nominal[b]=min(p['deliveries'][b] for tid in model.data['types']
                       if (p:=model.plan(tid,[b],[box['node']])) is not None)
    def priority(b):
        box=model.boxes[b]
        rule=settings.get('priority_rule','deadline')
        if rule=='target':
            return (min(box['deadline'],box['due']),not math.isfinite(box['deadline']),
                    -nominal[b]+settings['noise']*ranks[b],-box['weight'],b)
        if rule=='slack':
            return (box['deadline']-nominal[b],box['due'],-box['weight'],b)
        return (box['deadline'],box['due'],-box['weight']+settings['noise']*ranks[b],b)
    while remaining:
        seedbox=min(remaining,key=priority)
        proposals=[]
        for p in candidate_plans(model,sorted(remaining),seedbox,settings,rng):
            placed=place(model,state,p,settings)
            if placed:
                proposals.append(placed)
        if not proposals:
            return None,dict(reason='No feasible placement in the bounded candidate family',seed_box=seedbox,
                             scheduled_boxes=80-len(remaining),transport_sorties=len(state['trips']))
        proposals.sort(key=lambda p:p['score'])
        chosen=proposals[0]
        if settings.get('lookahead'):
            for proposal in proposals:
                trial=copy.deepcopy(state)
                commit(model,trial,proposal)
                future=remaining-set(proposal['plan']['box_ids'])
                representatives={}
                for bid in sorted(future):
                    b=model.boxes[bid]
                    if math.isfinite(b['deadline']):
                        old=representatives.get(b['node'])
                        if old is None or b['deadline']<model.boxes[old]['deadline']:
                            representatives[b['node']]=bid
                viable=True
                for node,bid in representatives.items():
                    found=False
                    for tid in model.data['types']:
                        solo=model.plan(tid,[bid],[node])
                        if solo is not None and place(model,trial,solo,settings) is not None:
                            found=True
                            break
                    if not found:
                        viable=False
                        break
                if viable:
                    chosen=proposal
                    break
        trip=commit(model,state,chosen)
        remaining.difference_update(trip['box_ids'])
    result=dict(trips=state['trips'],relay_trips=state['relay_trips'],config=settings,seed=seed)
    result['metrics']=metrics(result,model)
    return result,None


def make_communication(result,model):
    rows=[]
    relay_by_id={r['id']:r for r in result['relay_trips']}
    for trip in result['trips']:
        relative=copy.deepcopy(trip)
        relative['segments']=shift_segments(trip['segments'],-trip['start'])
        _,_,pieces=model.radio(relative)
        for p in pieces:
            direct=p['direct']
            relay=relay_by_id.get(trip['relay_trip_id'])
            rid=None if direct else relay['id']
            margin=p['margin'] if direct else min(p['support'][relay['site_id']],model.terrain.link_margin(relay['position'],model.gateway,'backhaul'))
            row=dict(trip_id=trip['id'],phase=p['phase'],t0=p['t0']+trip['start'],t1=p['t1']+trip['start'],
                     mode='direct' if direct else 'relay',relay_trip_id=rid,margin_db=margin,certified=True,
                     semantics='guaranteed_provider; runtime prefers available direct link')
            if rows and all(rows[-1][k]==row[k] for k in ['trip_id','phase','mode','relay_trip_id']) and abs(rows[-1]['t1']-row['t0'])<1e-6:
                rows[-1]['t1']=row['t1']
                rows[-1]['margin_db']=min(rows[-1]['margin_db'],row['margin_db'])
            else:
                rows.append(row)
    return rows


def solve(data,terrain,config=None):
    config=config or {}
    starts=config.get('starts',36)
    seed=config.get('seed',20260924)
    print('Generating relay candidates...',flush=True)
    sites=terrain.candidate_relay_positions(data['nodes'],step_m=config.get('relay_grid_m',500),agl_values=(150,300),max_candidates=40)
    print(f'Relay candidates: {len(sites)}',flush=True)
    model=Model(data,terrain,sites,spatial_step=config.get('spatial_step_m',100))
    history,solutions=[],[]
    for i in range(starts):
        settings=dict(max_stops=[1,2,3][i%3],neighbors=4,plan_limit=8,
                      packing_power=[0.8,1.1,1.35][(i//3)%3],
                      energy_weight=[80,160,40][i%3],wait_weight=0.5,
                      relay_weight=[250,500,100][i%3],noise=0 if i<3 else 6,
                      hard_first=i>=9 and i%2==1)
        if i>=12:
            j=i-12
            settings.update(priority_rule='target' if j%4<3 else 'slack',
                            max_stops=[1,2,3][j%3],hard_first=False,
                            packing_power=[0.7,0.9,1.1,1.3][(j//3)%4],
                            hard_bonus=0 if j<12 else 1,
                            energy_weight=[20,80,160][j%3],
                            relay_weight=[80,250,500][j%3],noise=0 if j<12 else 30)
            settings.update(lookahead=True,coverage_weight=[0,80,160][j%3])
        begun=time.perf_counter()
        print(f'Search {i+1}/{starts}: stops={settings["max_stops"]}',flush=True)
        result,failure=construct(model,settings,seed+i)
        if result is None:
            row=dict(label=f'S{i+1:02d}',feasible=False,**failure)
            print(f'  rejected: {failure}',flush=True)
        else:
            row=dict(label=f'S{i+1:02d}',feasible=True,**result['metrics'])
            solutions.append(result)
            print(f'  {row["transport_sorties"]} transport / {row["relay_sorties"]} relay, {row["makespan_s"]/60:.2f} min, {row["total_energy_kwh"]:.3f} kWh',flush=True)
        row['runtime_s']=time.perf_counter()-begun
        history.append(row)
    if not solutions:
        raise RuntimeError(f'No feasible complete schedule found: {history}')
    def objective(r):
        m=r['metrics']
        return (m['weighted_tardiness'],m['makespan_s'],m['total_energy_kwh'],m['transport_sorties']+m['relay_sorties'])
    solutions.sort(key=objective)
    refinement_budget=config.get('refine_seconds',90.0)
    if refinement_budget>0:
        from .refine import refine_schedule
        templates=[]
        for candidate in [solutions[0],min(solutions,key=lambda r:r['metrics']['total_energy_kwh']),
                          min(solutions,key=lambda r:r['metrics']['transport_sorties'])]:
            if not any(candidate is previous for previous in templates):
                templates.append(candidate)
        for j,template in enumerate(templates):
            print(f'Constraint scheduling refinement {j+1}/{len(templates)}...',flush=True)
            refined=refine_schedule(template,data,model,time_limit_s=refinement_budget/len(templates),seed=seed+j)
            if refined is not None:
                refined['metrics']=metrics(refined,model)
                history.append(dict(label=f'CP{j+1}',feasible=True,**refined['metrics'],
                                    refinement=refined.get('refinement',{})))
                solutions.append(refined)
                m=refined['metrics']
                print(f'  {m["makespan_s"]/60:.2f} min; weighted delay {m["weighted_tardiness"]:.2f}; {m["total_energy_kwh"]:.3f} kWh',flush=True)
        solutions.sort(key=objective)
    from .verify import verify
    for result in solutions:
        result['communication']=make_communication(result,model)
        print('Independent continuous and resource validation...',flush=True)
        validation=verify(result,data,terrain)
        if validation['passed']:
            result.update(validation=validation,search_history=history,relay_candidates=sites,
                          candidate_search=getattr(terrain,'candidate_search',{}),terrain=terrain.metadata(),
                          search_bounds=dict(starts=starts,seed=seed,relay_grid_m=config.get('relay_grid_m',500),
                              spatial_step_m=config.get('spatial_step_m',100),max_stops=3,
                              relay_providers_per_transport_sortie=1,refinement_total_seconds=refinement_budget,
                              global_optimality_proven=False))
            return result
        print(f'Validation rejected schedule: {validation["errors"][:4]}',flush=True)
    raise RuntimeError('All complete search schedules failed independent verification')
