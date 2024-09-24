project('pyoculus', 'c', version : '1.0.01',  meson_version: '>= 0.64.0',
)

add_languages('fortran', native: false)

fortran_compiler = meson.get_compiler('fortran')


f90flags = []
if fortran_compiler.get_id() == 'gcc'
    f90flags += '-ffree-line-length-none'
endif

f90flags += '-O3'
add_project_arguments(f90flags, language: 'fortran')

py3 = import('python').find_installation(pure: false)

sourcefiles = files(
    'pyoculus/maps/SPECfortran/pyvariables.f90',
    'pyoculus/maps/SPECfortran/pybasefn.f90',
    'pyoculus/maps/SPECfortran/pycoords.f90',
    'pyoculus/maps/SPECfortran/pybfield.f90',
    'pyoculus/maps/SPECfortran/pyPJH.f90'
)

module_name = 'pyoculus_spec_fortran_module'

# Get Python version and system info
python_version = py3.language_version().split('.')
py_suffix = 'cpython-@0@@1@-@2@-@3@'.format(python_version[0], python_version[1], host_machine.cpu_family(), host_machine.system())

if host_machine.system() == 'darwin'
    py_suffix = 'cpython-@0@@1@-darwin'.format(python_version[0], python_version[1])
    ext = '.so'
elif host_machine.system() == 'windows'
    py_suffix = 'cp@0@@1@-win_amd64'.format(python_version[0], python_version[1])
    ext = '.pyd'
elif host_machine.system() == 'linux'
    py_suffix = 'cpython-@0@@1@-@2@-linux-gnu'.format(python_version[0], python_version[1], host_machine.cpu_family())
    ext = '.so'
else
    py_suffix = ''
    ext = '.so'  # Default to .so for unknown systems
endif

output_name = module_name+ '.' + py_suffix + ext
message('trying to generate a compiled library called: ' + output_name)


f2py_target = custom_target(module_name,
    input : sourcefiles,
    output : output_name,
    command : [
        py3,
        '-m', 'numpy.f2py',
        '-c',
        '-m', module_name,
        '@INPUT@',
        '--opt=' + ' '.join(f90flags),
    ],
    depend_files : sourcefiles,
    install : true,
    install_dir : py3.get_install_dir() 
)


py3.install_sources(
   'pyoculus/__init__.py',
    subdir: 'pyoculus' 
    )

py3.install_sources(
  'pyoculus/fields/__init__.py',
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
  subdir: 'pyoculus/fields'
)

py3.install_sources(
  'pyoculus/geo/interpolate_coordinates.py',
  'pyoculus/geo/__init__.py',
  subdir: 'pyoculus/geo'
)

py3.install_sources(
  'pyoculus/integrators/base_integrator.py',
  'pyoculus/integrators/scipyode_integrator.py',
  'pyoculus/integrators/sdc_integrator.py',
  'pyoculus/integrators/__init__.py',
  subdir: 'pyoculus/integrators'
)

py3.install_sources(
  'pyoculus/maps/base_map.py',
  'pyoculus/maps/cylindrical_bfield_section.py',
  'pyoculus/maps/integrated_map.py',
  'pyoculus/maps/spec_pjh.py',
  'pyoculus/maps/spec_problem.py',
  'pyoculus/maps/toroidal_bfield_section.py',
  'pyoculus/maps/__init__.py',
  subdir: 'pyoculus/maps'
)

py3.install_sources(
  'pyoculus/solvers/base_solver.py',
  'pyoculus/solvers/fixed_point.py',
  'pyoculus/solvers/flux_surface_gr.py',
  'pyoculus/solvers/lyapunov_exponent.py',
  'pyoculus/solvers/manifold.py',
  'pyoculus/solvers/poincare_plot.py',
  'pyoculus/solvers/qfm.py',
  'pyoculus/solvers/__init__.py',
  subdir: 'pyoculus/solvers'
)

py3.install_sources(
  'pyoculus/toybox/cylindrical_toybox.py',
  'pyoculus/toybox/__init__.py',
  subdir: 'pyoculus/toybox'
)

py3.install_sources(
  'pyoculus/utils/continued_fraction.py',
  'pyoculus/utils/plot.py',
  'pyoculus/utils/cyl_cart_transform.py',
  'pyoculus/utils/convergence_domain.py',
  'pyoculus/utils/__init__.py',
  subdir: 'pyoculus/utils'
)

# Verify the build
#run_command(py3, '-c', 'import ' + module_name, check: true)
