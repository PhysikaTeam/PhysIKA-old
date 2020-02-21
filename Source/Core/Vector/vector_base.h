#pragma once
#include <iostream>
#include "Core/Platform.h"

namespace PhysIKA {
	template <typename Scalar, int Dim>
	class Vector
	{
	public:
		COMM_FUNC Vector() {};
		COMM_FUNC ~Vector() {};
	};

}  //end of namespace PhysIKA

