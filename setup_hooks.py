def annotate(cmd):
    '''Tell compiler to generate annotated HTMLs of Cython code'''
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True


def include_numpy(cmd):
    '''Add numpy include dir for all extensions, see setup.cfg'''
    import numpy
    for extension in cmd.extensions:
        extension.include_dirs.append(numpy.get_include())


def maybe_enable_openmp(cmd):
    '''Tell compiler to use OpenMP if OMP_USE is 1'''
    # Should be a compiler check, not a platform check...
    import os
    if os.environ.get('OMP_USE') == '1':
        cmd.announce('enabling OpenMP and breaking manylinux1 compatibility')
        import platform
        for extension in cmd.extensions:
            if platform.system() == 'Windows':
                extension.extra_compile_args.append('/openmp')
                extension.extra_link_args.append('/openmp')
            else:
                extension.extra_compile_args.append('-fopenmp')
                extension.extra_link_args.append('-fopenmp')
    else:
        cmd.announce('not enabling OpenMP and staying manylinux1-compatible')



def optimize(cmd):
    '''Tell compiler to use optimizations, see setup.cfg'''
    # Should be a compiler check, not a platform check...
    import platform
    if platform.system() == 'Linux':
        for extension in cmd.extensions:
            extension.extra_compile_args.append('-O2')
            extension.extra_compile_args.append('-g0')


def optimize_native(cmd):
    '''Tell compiler to use machine-specific optimizations, see setup.cfg'''
    # Should be a compiler check, not a platform check...
    import platform
    if platform.system() == 'Linux':
        for extension in cmd.extensions:
            extension.extra_compile_args.append('-Ofast')
            extension.extra_compile_args.append('-march=native')


def preprocess_with_mako(cmd):
    '''Preprocess *.mako.pyx with mako before cythoning and compilation'''
    import os
    import distutils.log
    import mako.template
    for extension in cmd.extensions:
        for source_path in extension.sources:
            if not source_path.endswith('.preprocessed.pyx'):
                # This '.pyx' file is ready for compilation, skip preprocessing
                cmd.announce('not preprocessing %s with mako' % source_path,
                             level=distutils.log.DEBUG)
                continue
            # Source filename ends with 'preprocessed.pyx', that means it
            # must be first generated from a corresponding '.mako.pyx' file
            mako_path = source_path.replace('.preprocessed.pyx', '.mako.pyx')
            if os.path.exists(source_path):
                if os.path.getmtime(mako_path) < os.path.getmtime(source_path):
                    # Preprocessed file is newer than source, skip
                    cmd.announce('skipping preprocessing %s with mako' %
                                 source_path,
                                 level=distutils.log.DEBUG)
                    continue
            cmd.announce('preprocessing %s with mako' % mako_path,
                         level=distutils.log.INFO)
            with open(mako_path, encoding='utf-8') as f:
                mako_source = f.read()
            preprocessed_source = mako.template.Template(mako_source).render()
            with open(source_path, 'w', encoding='utf-8') as f:
                f.write(preprocessed_source)
