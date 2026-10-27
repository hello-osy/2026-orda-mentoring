"""Recreate cad_paths.json from the bundled source drawing (developer utility)."""
import json
from pathlib import Path
import ezdxf
from ezdxf.path import make_path


def rebuild():
    assets = Path(__file__).resolve().parents[1] / 'orda_sim/assets'
    document = ezdxf.readfile(assets / 'track_drawing.dxf')
    origin = [128.493923, 121.986082]
    scale = .03
    paths = []
    for entity in document.modelspace().query('*[layer=="Visible"]'):
        points = [[(v.x - origin[0]) * scale, (v.y - origin[1]) * scale]
                  for v in make_path(entity).flattening(.025)]
        paths.append({'handle': entity.dxf.handle, 'type': entity.dxftype(), 'points': points})
    output = {'source': '2026Track 도면 시트1.dxf', 'cad_origin': origin,
              'meters_per_cad_unit': scale, 'extent_m': [10, 6], 'paths': paths}
    (assets / 'cad_paths.json').write_text(json.dumps(output, separators=(',', ':')))


if __name__ == '__main__':
    rebuild()
