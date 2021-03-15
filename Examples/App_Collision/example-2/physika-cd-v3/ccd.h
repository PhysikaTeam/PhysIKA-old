#pragma once

double
Intersect_EE(const vec3f &ta0, const vec3f &tb0, const vec3f &tc0, const vec3f &td0,
			 const vec3f &ta1, const vec3f &tb1, const vec3f &tc1, const vec3f &td1,
			 vec3f &qi);
double
Intersect_VF(const vec3f &ta0, const vec3f &tb0, const vec3f &tc0,
			 const vec3f &ta1, const vec3f &tb1, const vec3f &tc1,
			 const vec3f &q0, const vec3f &q1, vec3f &qi, vec3f &baryc);
