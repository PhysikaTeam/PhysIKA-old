#################################################################
# Scons script for Physika
# @author: Fei Zhu, 04/14/2014
# Usage: enter root directory of the project in terminal and
#        enter "scons"                
#################################################################

######################CONFIGURATIONS#############################
#BUILD TYPE
#build_type='Release'
build_type='Debug'

#BUILD MSVC PROJECTS FOR WINDOWS
build_msvc=True
#build_msvc=False
#################################################################

#IMPORTS
import fnmatch
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
os_name=platform.system()
os_architecture=platform.architecture()[0]

#SRC PATH
src_root_path='Physika_Src/'

#LIB NAMES
lib_names=get_immediate_subdirectories(src_root_path)
if 'Physika_Dependency' in lib_names: 
   lib_names.remove('Physika_Dependency')

#COMPILER
compiler=[]
if os_name in ('Linux','Darwin') or (os_name=='Windows' and build_msvc==False):
   compiler=['g++']
else:
   compiler=['msvc']

#BUILDERS
if build_type=='Release':
   compile=Builder(action='g++ -o $TARGET $SOURCE -c -O3 -fno-strict-aliasing -std=gnu++0x -DNDEBUG -I '+src_root_path)
else:
   compile=Builder(action='g++ -o $TARGET $SOURCE -c -g -fno-strict-aliasing -std=gnu++0x -I '+src_root_path)
arc_lib=Builder(action='ar rcs $TARGET $SOURCES')

#ENVIRONMENT
ENV={'PATH':os.environ['PATH']}
env=Environment(ENV=ENV)
if compiler==['g++']:
   env.Append(BUILDERS={'COMPILE':compile})
   env.Append(BUILDERS={'ARCLIB':arc_lib})
   env.Append(tools=['gcc','g++'])
else:
   env.Append(CPPPATH=src_root_path)
   if os_architecture=='32bit':
   	env.Append(MSVS_ARCH='x86')
   else:
	env.Append(MSVS_ARCH='amd64')
   if build_type=='Relase':
      env.Append(CCFLAGS=['/Ox','/EHsc','/DNDEBUG'])
   else:
      env.Append(CCFLAGS=['/Zi','/EHsc'])

#LIB PREFIX AND SUFFIX 
if os_name=='Windows':
    obj_suffix='.obj'
    lib_suffix='.lib'
elif os_name in ('Linux','Darwin'):
    lib_preffix='lib'
    obj_suffix='.o'
    lib_suffix='.a'

#IGNORED SRC FILES
#TO EXCLUDE FILES THAT ARE INCOMPLETE YET
ignored_src_files=['Physika_Src/Physika_Core/Matrices/sparse_matrix.cpp']

#COMPILE SRC FILES AND ARCHIVE INTO LIBS, GENERATE MSVC PROJECTS OPTIONALLY
header_files=[]
lib_files=[]
proj_files=[]
for name in lib_names:
    lib_obj_files=[]
    lib_src_files=[]
    lib_header_files=[]
    dir_path=os.path.join(src_root_path,name)
    for dir,_,_ in os.walk(dir_path):
    	lib_src_files.extend(glob(os.path.join(dir,'*.cpp')))
	lib_header_files.extend(glob(os.path.join(dir,'*.h')))
    	header_files.extend(glob(os.path.join(dir,'*.h')))
    if compiler==['g++']:
       for src_file in lib_src_files:
    	   if src_file not in ignored_src_files:
    	      obj_file=os.path.splitext(src_file)[0]+obj_suffix
	      lib_obj_files.append(obj_file)
	      env.COMPILE(obj_file,src_file)
    lib_file=name+lib_suffix
    if os_name in ('Linux','Darwin'):
       lib_file=lib_preffix+lib_file
    if compiler==['g++']:
       env.ARCLIB(lib_file,lib_obj_files)
    else:
       lib=env.StaticLibrary(target=lib_file,source=lib_src_files)
       proj=env.MSVSProject(target=name+env['MSVSPROJECTSUFFIX'],srcs=lib_src_files,incs=lib_header_files,buildtarget=lib,variant=build_type,auto_build_solution=0)
       proj_files.append(str(proj[0]))
    lib_files.append(lib_file)
#GENERATE MSVC SOLUTION
sln=[]
if compiler==['msvc']:
   sln=env.MSVSSolution(target='Physika'+env['MSVSSOLUTIONSUFFIX'],projects=proj_files,variant=build_type)

#COPY HEADERS AND LIB FILES TO TARGET DIRECTORY
header_target=[]
target_root_path='Public_Library/'
for header_file in header_files:
    if header_file.find(src_root_path)==0:
       target_file=header_file.replace(src_root_path,target_root_path+'include/')
       header_target=Command(target_file,header_file,Copy("$TARGET","$SOURCE"))
for lib_file in lib_files:
    target_file=target_root_path+'lib/'+os.path.basename(lib_file)
    Command(target_file,lib_file,Move("$TARGET","$SOURCE"))

#COPY DEPENDENCIES
src_dependency_root_path=src_root_path+'Physika_Dependency/'
pure_copy=['Eigen']
dependencies=get_immediate_subdirectories(src_dependency_root_path)
target_dependency_include_path=target_root_path+'include/Physika_Dependency/'
for name in dependencies:
    if name in pure_copy: #ONLY HEADERS
       Command(target_dependency_include_path+name,src_dependency_root_path+name,Copy("$TARGET","$SOURCE"))
    else:
       #COPY HEADERS
       Command(target_dependency_include_path+name+'/',src_dependency_root_path+name+'/include/',Copy("$TARGET","$SOURCE"))
       #COPY LIBS
       src_dependency_lib_path=src_dependency_root_path+name+'/lib/'
       if os_name=='Linux':
       	  src_dependency_lib_path=src_dependency_lib_path+'Linux/'
       elif os_name=='Darwin':
       	  src_dependency_lib_path=src_dependency_lib_path+'Apple/'
       elif os_name=='Windows':
       	  src_dependency_lib_path=src_dependency_lib_path+'Windows/'
       if os_architecture=='32bit':
       	  src_dependency_lib_path=src_dependency_lib_path+'X86/'
       else:
	  src_dependency_lib_path=src_dependency_lib_path+'X64/'
       for lib_name in os.listdir(src_dependency_lib_path):
       	  Command(target_root_path+'lib/'+lib_name,os.path.join(src_dependency_lib_path,lib_name),Copy("$TARGET","$SOURCE")) 

#CUSTOMIZE CLEAN OPTION
sln_delete_files=[build_type+'/','obj/','Physika.suo','Physika.sdf']
for name in os.listdir('./'):
    if name.endswith('.user') or name.endswith('.pdb'):
        sln_delete_files.append(name)
header_delete_files= [os.path.join(target_root_path+'include/', name) for name in os.listdir(target_root_path+'include/')
                      if os.path.isdir(os.path.join(target_root_path+'include/', name))]
Clean(sln,sln_delete_files)
Clean(header_target,header_delete_files)


    
    	    
