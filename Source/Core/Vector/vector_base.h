#pragma once
#include <iostream>
#include "Core/Platform.h"

namespace PhysIKA {

	template<typename Scalar>
	class VectorBase
	{
	public:
		virtual int size()const = 0;
		virtual Scalar& operator[] (unsigned int) = 0;
		virtual const Scalar& operator[] (unsigned int) const = 0;
	};


	template <typename Scalar, int Dim>
	class Vector
	{
	public:
		COMM_FUNC Vector() {};
		COMM_FUNC ~Vector() {};
	};

}  //end of namespace PhysIKA

