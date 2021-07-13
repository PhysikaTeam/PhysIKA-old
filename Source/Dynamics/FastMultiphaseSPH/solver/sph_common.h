#pragma once

#include "../math/geometry.h"

#define TYPE_FLUID 0

#define TYPE_RIGID 4
#define TYPE_DEFORMABLE 5
#define TYPE_GRANULAR 6

#define TYPE_NULL 99

#define NUM_NEIGHBOR 45

#define GROUP_FIXED -1
#define GROUP_MOVABLE 0

/* Yield criterion for elasto-plastic solids.
*/
#define VON_MISES 1
#define DRUCKER_PRAGER 2