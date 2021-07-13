#pragma once
#include "Dynamics/ParticleSystem/ElastoplasticityModule.h"

namespace PhysIKA {

template <typename TDataType>
class SummationDensity;

template <typename TDataType>
class FractureModule : public ElastoplasticityModule<TDataType>
{
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;
    typedef TPair<TDataType>          NPair;

    FractureModule();
    ~FractureModule() override{};

    void applyPlasticity() override;
};

#ifdef PRECISION_FLOAT
template class FractureModule<DataType3f>;
#else
template class FractureModule<DataType3d>;
#endif
}  // namespace PhysIKA