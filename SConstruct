#################################################################
# Scons script for Physika
# @author: Fei Zhu, 04/14/2014
#          Wei Chen,04/07/2016
# Usage: enter root directory of the project in terminal and
#        enter "scons"                
#################################################################

######################CONFIGURATIONS#############################

#MSVC VERSION FOR WINDOWS ENV, TO SUPPORT C++11/14, VS2015 IS NEEDED
msvc_version = '14.0'   #VS2015
#msvc_version = '14.1'  #VS2017

#USE OPENMP
#use_openmp = True
use_openmp = False

#USE CUDA
use_cuda = True
#use_cuda = False
#################################################################


#################################################################
#HINTS OUTPUT

print(
'''
*********************************************************************************************
Note:
1. To support C++11/14 on windows platform, VS2015 or VS2017 is needed, you would sepcify "msvc_version" variable in this script.
2. You would also specify to use "openmp" and enable "cuda" compiling by setting the "use_openmp" & "use_cuda" variable in this script.
3. When disable cuda, we would ignore all .cu files, and all .cpp files that if their path exists a string name "cuda" or "gpu"(case insensitive). 

Trouble Shooting:
1. On windows 8/10, administrator privilege is required to "scons" the project, so you should run shell window as admin.
   For the same reason, you should also open .sln file as admin to enable building in VS IDE.
*********************************************************************************************
''')

import SCons.Tool.MSCommon.vs as vs
print("*********************************************************************************************")
print("Your installed MSVC Version:")
print(vs.query_versions())
print("*********************************************************************************************")

#################################################################

#IMPORTS
import os
import platform
from os.path import basename
from glob import glob

#DEFS
#####################################################
def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]
#####################################################

#OS TYPE
os_name         = platform.system()
os_architecture = platform.architecture()[0]

#SRC PATH
src_root_path = 'Physika_Src/'

#INCLUDE_PATH
gl_include_path   = src_root_path+'Physika_Dependency/OpenGL/'    #SPECIAL HANDLING FOR OPENGL HEADERS, TO KEEP THE "#include <GL/XXX.h>" STYLE
glui_include_path = src_root_path+'Physika_Dependency/glui/'
include_path      = [src_root_path, gl_include_path, glui_include_path]

#LIB NAMES
lib_names = get_immediate_subdirectories(src_root_path)
if 'Physika_Dependency' in lib_names: 
    lib_names.remove('Physika_Dependency')
    
#LIB PREFIX AND SUFFIX 
if os_name == 'Windows':
    obj_suffix = '.obj'
    lib_suffix = '.lib'
elif os_name in ('Linux', 'Darwin'):
    lib_preffix = 'lib'
    obj_suffix = '.o'
    lib_suffix = '.a'

#COMPILER
compiler = ''
if os_name in ('Linux', 'Darwin'):
    compiler = 'g++'
else:
    compiler = 'msvc'

#ENVIRONMENT
ENV = {'PATH':os.environ['PATH']}
if os_architecture == '32bit':
    arc = 'x86'
else:
    arc = 'amd64'    
env = Environment(ENV = ENV, CPPPATH = include_path, MSVS_ARCH = arc, TARGET_ARCH = arc, MSVC_VERSION = msvc_version)

#IGNORED SRC FILES
#TO EXCLUDE FILES THAT ARE INCOMPLETE YET
ignored_src_files = []

#COMPILE SRC FILES AND ARCHIVE INTO LIBS, GENERATE MSVC PROJECTS OPTIONALLY
header_files = []
proj_files  = []
target_root_path = 'Public_Library/'

#SEPARATE RELEASE ENV & DEBUG ENV
release_env = env.Clone()
debug_env = env.Clone()
if compiler == 'g++':
    #release
    CCFLAGS = ['-O3', '-Wall', '-fno-strict-aliasing', '-std=gnu++11', '-DNDEBUG']
    if use_openmp: CCFLAGS.append('-fopenmp') #openmp support
    release_env.Replace(CCFLAGS = CCFLAGS)
        
    #debug
    CCFLAGS = ['-g', '-Wall', '-fno-strict-aliasing', '-std=gnu++11']
    if use_openmp: CCFLAGS.append('-fopenmp') #openmp support
    debug_env.Replace(CCFLAGS = CCFLAGS)
else:
    #release
    CCFLAGS = ['/Ox', '/EHsc', '/DNDEBUG', '/W3', '/MDd', '/wd\"4819\"']
    if use_openmp: CCFLAGS.append('/openmp') #openmp support
    release_env.Replace(CCFLAGS = CCFLAGS)
        
    #debug
    CCFLAGS = ['/Od', '/Zi', '/EHsc', '/W3', '/MDd', '/wd\"4819\"']
    if use_openmp: CCFLAGS.append('/openmp') #openmp support
    debug_env.Replace(CCFLAGS = CCFLAGS)

#CUDA SUPPORT
if use_cuda == True:
    release_env.Tool('cuda', toolpath = ['./Documentation/Cuda_Scons_Tool/'])
    debug_env.Tool('cuda', toolpath = ['./Documentation/Cuda_Scons_Tool/'])
    
    NVCCLINKCOM = '$NVCC $CUDA_ARCH -dlink $SOURCES -lcurand -lcudadevrt -o $TARGET'
    cuda_link_builder = Builder(action = NVCCLINKCOM)
    release_env.Append(BUILDERS = {'cuda_linker' : cuda_link_builder})
    debug_env.Append(BUILDERS = {'cuda_linker' : cuda_link_builder})

#COMPILE LIBS
for name in lib_names:
    lib_src_files = []
    lib_header_files = []
    dir_path = os.path.join(src_root_path, name)
    
    for dir,_,_ in os.walk(dir_path):
        lib_src_files.extend(glob(os.path.join(dir, '*.cpp')))
        lib_header_files.extend(glob(os.path.join(dir, '*.h')))
        header_files.extend(glob(os.path.join(dir, '*.h')))
        
        if use_cuda == True:
            lib_src_files.extend(glob(os.path.join(dir, '*.cu')))
            lib_header_files.extend(glob(os.path.join(dir, '*.cuh')))
            header_files.extend(glob(os.path.join(dir, '*.cuh')))

    #disable compiling for .cpp file if not use cuda && file path exists a 'cuda' or 'gpu' string (case insensitive)
    if use_cuda == False:
        for src_file in lib_src_files:
            lower_src_file = src_file.lower()
            if lower_src_file.find('cuda') != -1 or lower_src_file.find('gpu') != -1:
                print('ignore cuda file: ' + src_file)
                ignored_src_files.append(src_file)

    #remove igonre src files
    lib_src_files = [src_file for src_file in lib_src_files if src_file not in ignored_src_files]

                
    lib_file = name+lib_suffix
    if os_name in ('Linux', 'Darwin'):
        lib_file = lib_preffix+lib_file
    release_lib_file = target_root_path+'lib/release/'+os.path.basename(lib_file)
    debug_lib_file   = target_root_path+'lib/debug/'+os.path.basename(lib_file)
    
    #============================================================================================================================================
    
    #release obj files
    release_obj_files = [os.path.splitext(src_file)[0]+'_release'+('_cu' if os.path.splitext(src_file)[1] == '.cu' else '')+obj_suffix  for src_file in lib_src_files]
    for obj_file, src_file in zip(release_obj_files, lib_src_files):
        release_env.Object(obj_file, src_file)
    
    #release lib
    release_lib = release_env.StaticLibrary(target = release_lib_file, source = release_obj_files)
    
    #============================================================================================================================================
    
    #debug obj files
    debug_obj_files = [os.path.splitext(src_file)[0]+'_debug'+('_cu' if os.path.splitext(src_file)[1] == '.cu' else '')+obj_suffix for src_file in lib_src_files]
    for obj_file, src_file in zip(debug_obj_files, lib_src_files):
        debug_env.Object(obj_file, src_file)
    
    #debug lib
    debug_lib = debug_env.StaticLibrary(target = debug_lib_file, source = debug_obj_files)
    
    #============================================================================================================================================
    
    if compiler == 'msvc':
        proj = debug_env.MSVSProject(target = name+env['MSVSPROJECTSUFFIX'], srcs = lib_src_files, incs = lib_header_files, buildtarget = debug_lib, variant = 'debug', auto_build_solution = 0)
        proj_files.append(str(proj[0]))

if use_cuda:
    all_cuda_src_files = []
    for dir,_,_ in os.walk(src_root_path):
        all_cuda_src_files.extend(glob(os.path.join(dir, '*.cu')))   
    
    #=======================================================================================================================
    all_release_cuda_obj_files = [os.path.splitext(src_file)[0]+'_release_cu'+obj_suffix for src_file in all_cuda_src_files]
    release_cuda_link_target = os.path.join(src_root_path, 'release_cuda_link.obj')
    release_cuda_link_obj = release_env.cuda_linker(target = release_cuda_link_target, source = all_release_cuda_obj_files)
    
    release_cuda_link_lib  = target_root_path+'lib/release/cuda_link'
    release_env.StaticLibrary(target = release_cuda_link_lib, source = release_cuda_link_obj)
    
    #=======================================================================================================================
    all_debug_cuda_obj_files = [os.path.splitext(src_file)[0]+'_debug_cu'+obj_suffix for src_file in all_cuda_src_files]
    debug_cuda_link_target = os.path.join(src_root_path, 'debug_cuda_link.obj')
    debug_cuda_link_obj = debug_env.cuda_linker(target = debug_cuda_link_target, source = all_debug_cuda_obj_files)
    
    debug_cuda_link_lib = target_root_path+'lib/debug/cuda_link'
    debug_env.StaticLibrary(target = debug_cuda_link_lib, source = debug_cuda_link_obj)
           
#GENERATE MSVC SOLUTION
sln = []
if compiler == 'msvc':
    sln = env.MSVSSolution(target = 'Physika'+env['MSVSSOLUTIONSUFFIX'], projects = proj_files, variant = 'debug')

#COPY HEADERS TO TARGET DIRECTORY, LIBS ARE ALREADY THERE
header_target = []
for header_file in header_files:
    if header_file.find(src_root_path) == 0:
        target_file = header_file.replace(src_root_path, target_root_path+'include/')
        header_target = Command(target_file, header_file, Copy("$TARGET", "$SOURCE"))

#COPY DEPENDENCIES
src_dependency_root_path = src_root_path+'Physika_Dependency/'
pure_copy = ['Eigen']
dependencies = get_immediate_subdirectories(src_dependency_root_path)
target_dependency_include_path = target_root_path+'include/Physika_Dependency/'

for name in dependencies:
    if name in pure_copy: 
        #ONLY HEADERS, DIRECT COPY
        Command(target_dependency_include_path+name, src_dependency_root_path+name, Copy("$TARGET","$SOURCE"))
    elif os.path.isfile(src_dependency_root_path+name+'/SConscript'):
        #SCONS SCRIPT FOR THE DEPENDENCY EXISTS, CALL THE BUILD
        SConscript(src_dependency_root_path+name+'/SConscript', exports = 'release_env os_name os_architecture compiler')
    else: 
        #LIBS ARE PRECOMPILED, COPY HEADERS AND LIBS RESPECTIVELY
        
        #COPY HEADERS
        if name == 'OpenGL':  #SPECIAL HANDLING FOR OPENGL HEADERS, TO KEEP THE "#include <GL/XXX.h>" STYLE
            lib_header_files = glob(os.path.join(src_dependency_root_path+name, 'GL/*.h'))
            
            #copy glm
            Command(target_dependency_include_path+name+'/glm', src_dependency_root_path+name+'/glm', Copy("$TARGET","$SOURCE"))
                
            for header_file in lib_header_files:
                target_file = header_file.replace(src_dependency_root_path, target_dependency_include_path)    #COPY INTO OPENGL HEADER DIRECTORY
                Command(target_file, header_file, Copy("$TARGET", "$SOURCE"))
        else:
            Command(target_dependency_include_path+name+'/', src_dependency_root_path+name+'/include/', Copy("$TARGET","$SOURCE"))
            
        #COPY LIBS
        src_dependency_lib_path = src_dependency_root_path+name+'/lib/'
        
        if os_name == 'Linux':
            src_dependency_lib_path = src_dependency_lib_path+'Linux/'
        elif os_name == 'Windows':
            src_dependency_lib_path = src_dependency_lib_path+'Windows/'
        elif os_name == 'Darwin':
            src_dependency_lib_path = src_dependency_lib_path+'Apple/'
            
        if os_architecture == '32bit':
            src_dependency_lib_path = src_dependency_lib_path+'X86/'
        else:
            src_dependency_lib_path = src_dependency_lib_path+'X64/'
            
        for lib_name in os.listdir(src_dependency_lib_path):
            Command(target_root_path+'lib/release/'+lib_name, os.path.join(src_dependency_lib_path, lib_name), Copy("$TARGET", "$SOURCE"))
            Command(target_root_path+'lib/debug/'+lib_name, os.path.join(src_dependency_lib_path, lib_name), Copy("$TARGET", "$SOURCE")) 

#CUSTOMIZE CLEAN OPTION
sln_delete_files = ['release/', 'obj/', 'Physika.suo', 'Physika.sdf']
sln_delete_files.extend(['debug/', 'obj/', 'Physika.suo', 'Physika.sdf'])
for name in os.listdir('./'):
    if name.endswith('.user') or name.endswith('.pdb') or name.endswith('.ilk') or name.endswith('.db') or name.endswith('.vs'):
        sln_delete_files.append(name)

header_delete_files = [os.path.join(target_root_path+'include/', name) for name in os.listdir(target_root_path+'include/')
                      if os.path.isdir(os.path.join(target_root_path+'include/', name))]
                      
Clean(sln, sln_delete_files)
Clean(header_target, header_delete_files)

cuda_link_objs = []
for dir,_,_ in os.walk(src_root_path):
        cuda_link_objs.extend(glob(os.path.join(dir, '*.link.obj')))
Clean(debug_lib, cuda_link_objs)

#DELETE ADDTIONAL LIB DIRECTORY
Clean(debug_lib,   target_root_path+'lib/debug/')
Clean(release_lib, target_root_path+'lib/release/')