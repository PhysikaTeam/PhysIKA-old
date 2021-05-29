#pragma once

#include "CollisionDate.h"
#include "Collision.h"

namespace PhysIKA {
void body_collide_gpu(
	std::vector<CollisionDate> bodys,
	std::vector<std::vector<TrianglePair>>& contacts,
	int& CCDtime,
	std::vector<ImpactInfo>& contact_info,
	float thickness);
}