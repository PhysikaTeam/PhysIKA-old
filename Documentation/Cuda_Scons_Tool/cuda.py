"""
SCons.Tool.cuda
@author: WeiChen, 07/02/2016
@breif: Tool for Scons used to support compiling of cuda code
@usage:
       1. this file is used in SConstruct script through codes like: 
            env.Tool('cuda', toolpath = 'documentation/Cuda_Scons_Tool/')
       2. you can also put this file to PYTHON_HOME/Lib/site-packages/scons-x.x.x/SCons/Tool
@reference:
           https://bitbucket.org/scons/scons/wiki/CudaTool
           https://github.com/bryancatanzaro/cuda-scons/blob/master/nvcc.py
"""

import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults
import os
import sys
import platform

#cuda suffix
cuda_suffix = '.cu'

# make a CUDAScanner for finding #includes
# cuda uses the c preprocessor, so we can use the CScanner
cuda_scanner = SCons.Scanner.C.CScanner()

def generate(env):
    
    os_name         = platform.system()
    os_architecture = platform.architecture()[0]
    
    #cuda path
    cuda_bin_path = ''
    cuda_inc_path = ''
    cuda_lib_path = ''
    cuda_dll_path = ''
    
    cuda_path = None
    if 'CUDA_PATH' in os.environ:
        cuda_path = os.environ['CUDA_PATH']
    elif 'CUDA_PATH' in env:
        cuda_path = env['CUDA_PATH']
    else:
        guess_path = [ '/usr/local/NVIDIA_CUDA_TOOLKIT',
                       '/usr/local/CUDA_TOOLKIT',
                       '/usr/local/cuda_toolkit',
                       '/usr/local/CUDA',
                       '/usr/local/cuda'
                     ]
                     
        for path in guess_path:
            if os.path.isdir(path):
                cuda_path = path
                break
  
    if cuda_path == None:
        sys.exit("Cannot find the CUDA_PATH. Please install CUDA OR add CUDA_PATH in your environment variables OR explictly specify env['CUDA_PATH']!")
    
    cuda_inc_path = cuda_path+'/include/'
    cuda_bin_path = cuda_path+'/bin/'
    
    cuda_version_str = os.path.basename(cuda_path)
    cuda_version_id = filter(str.isdigit, cuda_version_str)
    
    if os_name == 'Windows':
        if os_architecture == '32bit':
            cuda_lib_path = cuda_path+'/lib/Win32/'
            cuda_dll_path = cuda_path+'/bin/cudart32_'+cuda_version_id+'.dll'
        else:
            cuda_lib_path = cuda_path+'/lib/X64/'
            cuda_dll_path = cuda_path+'/bin/cudart64_'+cuda_version_id+'.dll'
    elif os_name == 'Linux':
        if os_architecture == '32bit':
            cuda_lib_path = cuda_path+'/lib/'
        else:
            cuda_lib_path = cuda_path+'/lib64/'
    elif os_name == 'Darwin':
        cuda_lib_path = cuda_path+'/lib/'    
     
    #add include path
    env.Append(CPPPATH = cuda_inc_path)
     
    #add cuda runtime libpath and lib    
    env.Append(LIBPATH = cuda_lib_path)
    env.Append(LIBS = 'cudart')
    env.Append(LIBS = 'cudadevrt')
    env.Append(LIBS = 'curand')
    env['CUDA_DLL_PATH'] = cuda_dll_path

    # "NVCC common command line"
    if not env.has_key('_NVCCCOMCOM'):
        # nvcc needs '-I' prepended before each include path, regardless of platform
        env['_NVCCWRAPCPPPATH']   = '${_concat("-I ", CPPPATH, "", __env__)}'
        # prepend -Xcompiler before each flag
        env['_NVCCWRAPCFLAGS']    = '${_concat("-Xcompiler ", CFLAGS,    "", __env__)}'
        env['_NVCCWRAPSHCFLAGS']  = '${_concat("-Xcompiler ", SHCFLAGS,  "", __env__)}'
        
        #special treatment for Darwin(Mac)
        #since clang could report an error if '-Xcompiler -std-gnu++11' is used
        #while g++ just report a warning
        if os_name == 'Darwin':
            DARWIN_CCFLAGS = env['CCFLAGS'][:]  #copy
            if '-std=gnu++11' in DARWIN_CCFLAGS:
                DARWIN_CCFLAGS.remove('-std=gnu++11') 
            env['DARWIN_CCFLAGS'] = DARWIN_CCFLAGS

            DARWIN_SHCCFLAGS = env['SHCCFLAGS'][:]  #copy
            if '-std=gnu++11' in DARWIN_SHCCFLAGS:
                DARWIN_SHCCFLAGS.remove('-std=gnu++11') 
            env['DARWIN_SHCCFLAGS'] = DARWIN_SHCCFLAGS

            env['_NVCCWRAPCCFLAGS']   = '${_concat("-Xcompiler ", DARWIN_CCFLAGS,   "", __env__)}'
            env['_NVCCWRAPSHCCFLAGS'] = '${_concat("-Xcompiler ", DARWIN_SHCCFLAGS, "", __env__)}'
        else:
            env['_NVCCWRAPCCFLAGS']   = '${_concat("-Xcompiler ", CCFLAGS,   "", __env__)}'
            env['_NVCCWRAPSHCCFLAGS'] = '${_concat("-Xcompiler ", SHCCFLAGS, "", __env__)}'
        
    # assemble the common command line
    env['_NVCCCOMCOM'] = '${_concat("-Xcompiler ", CPPFLAGS, "", __env__)} $_CPPDEFFLAGS $_NVCCWRAPCPPPATH'
    
    # set the include path, and pass both c compiler flags and c++ compiler flags
    env['NVCCFLAGS'] = SCons.Util.CLVar('')
    env['SHNVCCFLAGS'] = SCons.Util.CLVar('') + ' -shared'
  
    # set cuda complier
    env['NVCC'] = 'nvcc'
    env['SHNVCC'] = 'nvcc'
    
    # set cuda compute arch
    env['CUDA_ARCH'] = '-arch=compute_52'
  
    # 'NVCC Command'
    env['NVCCCOM']   = '$NVCC   -o $TARGET $CUDA_ARCH -dlink -c -dc -std=c++11 $NVCCFLAGS   $_NVCCWRAPCFLAGS   $_NVCCWRAPCCFLAGS   $_NVCCCOMCOM $SOURCES'
    env['SHNVCCCOM'] = '$SHNVCC -o $TARGET $CUDA_ARCH -dlink -c -dc -std=c++11 $SHNVCCFLAGS $_NVCCWRAPSHCFLAGS $_NVCCWRAPSHCCFLAGS $_NVCCCOMCOM $SOURCES'

    # create builders that make static & shared objects from .cu files
    static_obj_builder, shared_obj_builder = SCons.Tool.createObjBuilders(env)

    # Add this suffix to the list of things buildable by Object
    static_obj_builder.add_action(cuda_suffix, '$NVCCCOM') 
    shared_obj_builder.add_action(cuda_suffix, '$SHNVCCCOM')
    static_obj_builder.add_emitter(cuda_suffix, SCons.Defaults.StaticObjectEmitter)
    shared_obj_builder.add_emitter(cuda_suffix, SCons.Defaults.SharedObjectEmitter)

    # Add this suffix to the list of things scannable
    SCons.Tool.SourceFileScanner.add_scanner(cuda_suffix, cuda_scanner)
    
    # Prepend cuda_bin_path
    env.PrependENVPath('PATH', cuda_bin_path)

def exists(env):
    return env.Detect('nvcc')
