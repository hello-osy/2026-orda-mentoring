import json,yaml,math,time
from pathlib import Path
from orda_sim.answer import scan_sectors,road_gap,avoidance_steering
p=Path('/tmp/orda-lane-test/run40');c=yaml.safe_load((p/'config.yaml').read_text());lc=c['answer']['lidar'];robot=c['robot']
raw=json.loads((p/'scan.json').read_text());scan=scan_sectors(**raw,config=lc);scan['allow_edge_crossing']=False
heading=-.0015675340633240302;center=-.325510309001805
cosine=max(.2,math.cos(heading));desired=road_gap(scan,robot,heading,center,lc['road_default_bias'],lc)
lane=dict(valid=True,error=-.2560640770568062,forward=.65,heading=heading,center_fit=[0.,math.tan(heading),(center+desired)/cosine],fit_near=0.,fit_far=2.,road_memory=True,road_center=center)
print('points',len(scan['points']),'front',scan['front'],'desired',desired,'target intercept',lane['center_fit'][2])
results=[]
for samples in [33,65,129]:
 for distance in [.10,.25,.40,.55,.70,1.0]:
  cfg=dict(lc,planning_steering_samples=samples,planning_return_samples=samples,planning_turn_distance=distance)
  start=time.perf_counter();r=avoidance_steering(scan,lane,robot,cfg,.0325)
  row=dict(samples=samples,turn_distance=distance,result=r,ms=(time.perf_counter()-start)*1000);results.append(row);print(row,flush=True)
(p/'probe.json').write_text(json.dumps(dict(scan_summary={'points':len(scan['points']),'front':scan['front']},desired=desired,lane=lane,results=results),indent=2))
