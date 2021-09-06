#pragma once

#define ZERO 0.00001                   //!< vector zero
#define VOXEL_VOLUME_ZERO 0.01         //!< ignore the cell when the area  < VOXEL_VOLUME_ZERO*VOXEL_VOLUME
#define VTK_VOLUME_ZERO 0.01           //!< ignore the cell when the area  < VTK_VOLUME_ZERO*AVERAGE_VTK_VOLUME
#define VTK_VOLUME_ZERO_RELATION 0.01  //!< when relation error<VTK_VOLUME_ZERO_RELATION, vertex in cell
#define RELATION_STEP 0.01             //!< must <1, step of  adjusting vertex coord
