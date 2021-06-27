#!/bin/bash

mesh_name=bunny
mesh=mesh-level-2.vtk
max=50000
min=1000
sub_level=2

# uniform material
# ../../build/bin/gen_vox_mtrl mesh=$mesh type=hexs mode=000 maxv=5000 minv=5000 mtr_file=beamC.sub1-mtr-0.txt

# inhomo
i=1
for mode in 010 100 001 111; do
    echo 'mtr '$i
    outdir=${mesh_name}_sub${sub_level}
    mkdir -p $outdir
    
    ../../build/bin/gen_vox_mtrl mesh=$mesh type=hexs mode=$mode maxv=$max minv=$min mtr_file=${outdir}/${mesh_name}.sub${sub_level}-mtr-$i.txt outmesh=${outdir}/temp$i.vtk
    let i=i+1
done
