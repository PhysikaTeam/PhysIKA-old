#pragma once
#include<string>

int fluidEvaluation(std::string& oriCdfName, 
					std::string& oriShapeName, 
					std::string& tarCdfName, 
					std::string& tarShapeName,
					std::string& rootPath,
					int max_steps);