#################################################################
# Scons script for Physika
# @author: Fei Zhu, 04/14/2014
# Usage: enter root directory of the project in terminal and
#        enter "scons"                
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

#BUILD TYPE
#build_type='release'
build_type='debug'

#SRC PATH
src_root_path='Physika_Src/'

#LIB NAMES
lib_names=get_immediate_subdirectories(src_root_path)
if 'Physika_Dependency' in lib_names: 
   lib_names.remove('Physika_Dependency')

#BUILDERS
if build_type=='release':
   compile=Builder(action='g++ -o $TARGET $SOURCE -c -O3 -fno-strict-aliasing -std=gnu++0x -DNDEBUG -I '+src_root_path)
else:
   compile=Builder(action='g++ -o $TARGET $SOURCE -c -g -fno-strict-aliasing -std=gnu++0x -I '+src_root_path)
arc_lib=Builder(action='ar rcs $TARGET $SOURCES')

#ENVIRONMENT
ENV={'PATH':os.environ['PATH']}
env=Environment(ENV=ENV)
env.Append(BUILDERS={'COMPILE':compile})
env.Append(BUILDERS={'ARCLIB':arc_lib})
env.Append(tools=['gcc','g++'])

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

#COMPILE SRC FILES AND ARCHIVE INTO LIBS
header_files=[]
lib_files=[]
for name in lib_names:
    obj_files=[]
    src_files=[]
    dir_path=os.path.join(src_root_path,name)
    for dir,_,_ in os.walk(dir_path):
    	src_files.extend(glob(os.path.join(dir,'*.cpp')))
    	header_files.extend(glob(os.path.join(dir,'*.h')))
    for src_file in src_files:
    	if src_file not in ignored_src_files:
    	   obj_file=os.path.splitext(src_file)[0]+obj_suffix
	   obj_files.append(obj_file)
	   env.COMPILE(obj_file,src_file)
    lib_file=name+lib_suffix
    if os_name in ('Linux','Darwin'):
       lib_file=lib_preffix+lib_file
    env.ARCLIB(lib_file,obj_files)
    lib_files.append(lib_file)

#COPY HEADERS AND LIB FILES TO TARGET DIRECTORY
target_root_path='Public_Library/'
for header_file in header_files:
    if header_file.find(src_root_path)==0:
       target_file=header_file.replace(src_root_path,target_root_path+'include/')
       Command(target_file,header_file,Copy("$TARGET","$SOURCE"))
for lib_file in lib_files:
    target_file=target_root_path+'lib/'+os.path.basename(lib_file)
    Command(target_file,lib_file,Move("$TARGET","$SOURCE"))
    
    	    