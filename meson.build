project('pyoculus', 'c', version : '1.0.0',  meson_version: '>= 0.64.0',
)

add_languages('fortran', native: false)

fortran_compiler = meson.get_compiler('fortran')

add_project_arguments('-DFFLAGS=-ffree-line-length-none', language : 'c')

f90flags = []
if fortran_compiler.get_id() == 'gcc'
    f90flags += '-ffree-line-length-none'
endif

f90flags += '-O3'


py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

incdir_numpy = run_command(
    py3, '-c',
    'import numpy; print(numpy.get_include())',
    check: true,
).stdout().strip()

incdir_f2py = run_command(
    py3, '-c',
    'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())',
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)


fortranobject_c = incdir_f2py / 'fortranobject.c'
# Share this object across multiple modules.
fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  dependencies: py3_dep,
  include_directories: inc_np,
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: inc_np,
)

sourcefiles = files(
        'pyoculus/maps/SPECfortran/pyvariables.f90',
        'pyoculus/maps/SPECfortran/pybasefn.f90',
        'pyoculus/maps/SPECfortran/pycoords.f90',
        'pyoculus/maps/SPECfortran/pybfield.f90',
        'pyoculus/maps/SPECfortran/pyPJH.f90'
)

py3.install_sources(
  'pyoculus/__init__.py',
  'pyoculus/fields/cylindrical_bfield.py',
  'pyoculus/fields/cylindrical_bfield_analytic.py',
  'pyoculus/fields/m3dc1_bfield.py',
  'pyoculus/fields/magnetic_field.py',
  'pyoculus/fields/qfm_bfield.py',
  'pyoculus/fields/simsopt_bfield.py',
  'pyoculus/fields/spec_bfield.py',
  'pyoculus/fields/toroidal_bfield.py',
  'pyoculus/fields/two_waves.py',
  'pyoculus/fields/__init__.py',
  'pyoculus/geo/interpolate_coordinates.py',
  'pyoculus/geo/__init__.py',
  'pyoculus/integrators/base_integrator.py',
  'pyoculus/integrators/scipyode_integrator.py',
  'pyoculus/integrators/sdc_integrator.py',
  'pyoculus/integrators/__init__.py',
  'pyoculus/maps/base_map.py',
  'pyoculus/maps/cylindrical_bfield_section.py',
  'pyoculus/maps/integrated_map.py',
  'pyoculus/maps/spec_pjh.py',
  'pyoculus/maps/spec_problem.py',
  'pyoculus/maps/toroidal_bfield_section.py',
  'pyoculus/maps/__init__.py',
  'pyoculus/solvers/base_solver.py',
  'pyoculus/solvers/fixed_point.py',
  'pyoculus/solvers/flux_surface_gr.py',
  'pyoculus/solvers/lyapunov_exponent.py',
  'pyoculus/solvers/manifold.py',
  'pyoculus/solvers/poincare_plot.py',
  'pyoculus/solvers/qfm.py',
  'pyoculus/solvers/__init__.py',
  'pyoculus/toybox/cylindrical_toybox.py',
  'pyoculus/toybox/__init__.py',
  'pyoculus/utils/continued_fraction.py',
  'pyoculus/utils/plot.py',
  'pyoculus/utils/cyl_cart_transform.py',
  'pyoculus/utils/convergence_domain.py',
  'pyoculus/utils/__init__.py',
  subdir: 'pyoculus'
)

