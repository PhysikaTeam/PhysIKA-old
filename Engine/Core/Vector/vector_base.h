#pragma once
#include <iostream>
#include "Core/Platform.h"

namespace Physika {
	template <typename Scalar, int Dim>
	class Vector
	{
	public:
		COMM_FUNC Vector() {};
		COMM_FUNC ~Vector() {};
	};

}  //end of namespace Physika

