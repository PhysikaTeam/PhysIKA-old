#pragma once

#ifndef POINTSDFCONTACTDETECTOR_H
#define POINTSDFCONTACTDETECTOR_H

#include "Framework/Framework/DeclareModuleField.h"
#include "Core/Platform.h"
#include "Core/Vector/vector_3d.h"
#include "Framework/Topology/DistanceField3D.h"
#include "Core/DataTypes.h"
#include "Dynamics/RigidBody/ContactInfo.h"
#include "Core/Array/DynamicArray.h"
#include "Dynamics/RigidBody/PBDRigid/PBDBodyInfo.h"
#include <vector>

namespace PhysIKA {

//template class DistanceField3D<DataType3d>;

class PointSDFContactDetector
{
public:
    PointSDFContactDetector();
    ~PointSDFContactDetector() {}

    void compute(DeviceDArray<ContactInfo<double>>& contacts,
                 DeviceDArray<Vector3d>&            points,
                 DistanceField3D<DataType3f>&       sdf,
                 DeviceArray<PBDBodyInfo<double>>&  body,
                 int                                sdfif,
                 int                                beginidx = 0);

    //DEF_EMPTY_IN_VAR(SDFid, int, "SDFid");

    //DEF_EMPTY_IN_VAR(SDF, DistanceField3D<DataType3d>, "SDF");

    ///**
    //* @brief Particle position
    //*/
    ////DEF_EMPTY_IN_ARRAY(Position, Vector3d, DeviceType::GPU, "Particle position");
    //DEF_EMPTY_IN_ARRAY(Position, Vector3d, DeviceType::GPU, "Particle position");

    ////DEF_EMPTY_OUT_ARRAY(Contacts, ContactInfo<double>, DeviceType::GPU, "Contacts");

private:
};

class PointMultiSDFContactDetector
{
public:
    //PointMultiSDFContactDetector() {}

    void setSDFs(std::shared_ptr<std::vector<DistanceField3D<DataType3f>>> sdfs)
    {
        m_sdfs = sdfs;
    }
    std::shared_ptr<std::vector<DistanceField3D<DataType3f>>> getSDFs() const
    {
        return m_sdfs;
    }
    void addSDF(DistanceField3D<DataType3f>& sdf)
    {
        if (!m_sdfs)
            m_sdfs = std::make_shared<std::vector<DistanceField3D<DataType3f>>>();
        m_sdfs->push_back(sdf);
    }

    void compute();

public:
    DeviceDArray<ContactInfo<double>>* m_contacts = 0;

    DeviceArray<PBDBodyInfo<double>>* m_body = 0;

    DeviceDArray<Vector3d>* m_particlePos = 0;

private:
    std::shared_ptr<std::vector<DistanceField3D<DataType3f>>> m_sdfs;
    PointSDFContactDetector                                   m_singleSDFDetector;
};

}  // namespace PhysIKA
#endif  //POINTSDFCONTACTDETECTOR_H