import os
import setuptools
from catkin_pkg.python_setup import generate_distutils_setup


setup_py_dir = os.path.dirname(os.path.realpath(__file__))
need_files = []
datadir = "grasping_benchmarks"

hh = setup_py_dir + "/" + datadir

for root, dirs, files in os.walk(hh):
  for fn in files:
    ext = os.path.splitext(fn)[1][1:]
    if ext and ext in 'xml yaml ini'.split(
    ):
      fn = root + "/" + fn
      need_files.append(fn[1 + len(hh):])


setuptools.setup(
    name="grasping-benchmarks",
    version="0.0.0",
    author="elena rampone",
    author_email="elena.rampone@iit.it",
    packages=setuptools.find_packages(),
    package_data={'grasping_benchmarks': need_files},
    python_requires='>=3',
    install_requires=['scipy', 'numpy', 'pyyaml'],
)
