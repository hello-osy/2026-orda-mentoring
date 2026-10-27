import inspect,json,math,yaml,time
from pathlib import Path
import orda_sim.answer as answer
from orda_sim.answer import *
p=Path('/tmp/orda-lane-test/run40');c=yaml.safe_load((p/'config.yaml').read_text());lc=c['answer']['lidar'];robot=c['robot']
raw=json.loads((p/'scan.json').read_text());scan=scan_sectors(**raw,config=lc);scan['allow_edge_crossing']=False
lane=json.loads((p/'probe.json').read_text())['lane']
src=inspect.getsource(answer.avoidance_steering)
src=src.replace('    candidates = []','    candidates = []\n    clearances=[]')
src=src.replace('    first_distances =','    print("reference",reference_steering)\n    first_distances =',1)
src=src.replace('                if clearance < margin:', '                clearances.append((clearance,float(steering),float(next_steering)))\n                if clearance < margin:')
src=src.replace('    return None if not candidates else float(min(candidates)[1])','    print("top clearance", sorted(clearances,reverse=True)[:5], "feasible",len(candidates))\n    return None if not candidates else float(min(candidates)[1])')
for name,extra in [('original',[]),('reference_only',['reference_steering']),('fine_only',['-.01625','.01625']),('reference_and_fine',['reference_steering','-.01625','.01625'])]:
 text=src
 if extra:
  text=text.replace('np.linspace(-robot[\'max_steering\'], robot[\'max_steering\'], samples)', 'np.unique(np.r_[np.linspace(-robot[\'max_steering\'], robot[\'max_steering\'], samples),'+','.join(extra)+'])')
  text=text.replace('np.linspace(-robot[\'max_steering\'], robot[\'max_steering\'], return_samples)', 'np.unique(np.r_[np.linspace(-robot[\'max_steering\'], robot[\'max_steering\'], return_samples),'+','.join(extra)+'])')
 exec(text,globals());start=time.perf_counter();result=avoidance_steering(scan,lane,robot,lc,.0325)
 print(name,result,1000*(time.perf_counter()-start))
