#pragma once

#include<string>
#include"vector_types.h"

void transformParticleIntoGrid(const std::string& inputName, 
	const std::string& outputName, 
	int RESOLUTION, 
	float particleRadius);


void transformParticleIntoGrid(std::vector<float3>& particlePos,
	std::vector<float>& particleDensity,
	std::string& outputName,
	int RESOLUTION,
	float particleRadius);