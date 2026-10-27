from setuptools import setup
from pathlib import Path
assets = [('share/orda_sim/' + str(folder), [str(path) for path in folder.iterdir() if path.is_file()])
          for folder in [Path('assets')]]
setup(name='orda_sim', version='0.1.0', packages=['orda_sim'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/orda_sim']),
                  ('share/orda_sim', ['package.xml'])] + assets,
      install_requires=['setuptools'], zip_safe=False,
      maintainer='ORDA', maintainer_email='osy@example.invalid',
      description='Autorace mentoring simulator', license='Proprietary',
      entry_points={'console_scripts': ['orda=orda_sim.entry:main']})
