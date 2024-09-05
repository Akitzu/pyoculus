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

inc_np = include_directories(incdir_numpy)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# Share this object across multiple modules.
fortranobject_lib = static_library('_fortranobject',
  fortranobject_c,
  dependencies: py3_dep,
  include_directories: [inc_np, inc_f2py],
)
fortranobject_dep = declare_dependency(
  link_with: fortranobject_lib,
  include_directories: [inc_np, inc_f2py],
)

sourcefiles = files(
        'pyoculus/maps/SPECfortran/pyvariables.f90',
        'pyoculus/maps/SPECfortran/pybasefn.f90',
        'pyoculus/maps/SPECfortran/pycoords.f90',
        'pyoculus/maps/SPECfortran/pybfield.f90',
        'pyoculus/maps/SPECfortran/pyPJH.f90'
)

compiled_fortran = custom_target('pyoculus_spec_fortran_module',
  input : sourcefiles,
  output: 'pyoculus_spec_fortran_modulemodule.c',
  command : [py3, '-m', 'numpy.f2py',  '-m', 'pyoculus_spec_fortran_module', '@INPUT@'],
  install : true,
  install_dir : py3.get_install_dir()
)

py3.extension_module('pyoculus_spec_fortran_module', 
                    sourcefiles, compiled_fortran, 
                    dependencies: fortranobject_dep, 
                    install: true, 
                    fortran_args: f90flags)

py3.install_sources(
           'pyoculus/__init__.py',
           'pyoculus/fields/*.py',
           'pyoculus/geo/*.py',
           'pyoculus/integrators/*.py',
           'pyoculus/maps/*.py',
           'pyoculus/solvers/*.py',
           'pyoculus/toybox/*.py',
           'pyoculus/utils/*.py',
           subdir: 'pyoculus' 
)

