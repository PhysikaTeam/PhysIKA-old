#pragma once
#include "Framework/Framework/Node.h"

namespace Physika
{
	/*!
	*	\class	HeightField
	*	\brief	A height field node
	*/
	template<typename TDataType>
	class HeightField : public Node
	{
		DECLARE_CLASS_1(HeightField, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HeightField();
		virtual ~HeightField();

	public:
		bool initialize() override;

	private:
	};


#ifdef PRECISION_FLOAT
	template class HeightField<DataType2f>;
#else
	template class HeightField<DataType2d>;
#endif
}