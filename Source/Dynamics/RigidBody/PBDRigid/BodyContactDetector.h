#pragma once

#ifndef PHYSIKA_HEIGHTFIELDBODYDETECTOR_H
#define PHYSIKA_HEIGHTFIELDBODYDETECTOR_H

#include "Framework/Framework/CollisionModel.h"
#include "Dynamics/RigidBody/RigidBody2.h"
#include "Dynamics/HeightField/HeightFieldGrid.h"
#include "Dynamics/RigidBody/ContactInfo.h"

#include "Framework/Topology/Primitive3D.h"

namespace PhysIKA
{

    class HeightFieldBodyDetector:public CollisionModel
    {
    public:


        virtual bool isSupport(std::shared_ptr<CollidableObject> obj) { return true; };


        virtual void doCollision();

        virtual void addCollidableObject(RigidBody2_ptr obj) { m_allBodies.push_back(obj); }


        DEF_EMPTY_VAR(Contacts, DeviceDArray<ContactInfo<double>>, "Contact information");

        DEF_EMPTY_VAR(Threshold, double, "Detection threshold");

    public:
        

    public:
        
        DeviceHeightField1d* m_land = 0;
        
        std::vector<RigidBody2_ptr> m_allBodies;

        DeviceDArray<int> m_counter;

    };


    class OrientedBodyDetector :public CollisionModel
    {
    public:


        virtual bool isSupport(std::shared_ptr<CollidableObject> obj) { return true; };


        virtual void doCollision();

        virtual void addCollidableObject(RigidBody2_ptr obj, std::shared_ptr<TOrientedBox3D<float>> obb) { 
            m_allBodies.push_back(obj); 
            m_obbs.push_back(obb);
        }


        DEF_EMPTY_VAR(Contacts, DeviceDArray<ContactInfo<double>>, "Contact information");

        DEF_EMPTY_VAR(Threshold, double, "Detection threshold");

    public:


    public:
        std::vector<std::shared_ptr<TOrientedBox3D<float>>> m_obbs;            // initial oriented bounding box.
        std::vector<std::shared_ptr<TOrientedBox3D<float>>> m_transformedObbs;


        std::vector<RigidBody2_ptr> m_allBodies;

        DeviceDArray<int> m_counter;

        HostDArray<ContactInfo<double>> m_hostContacts;
    };

}



#endif // PHYSIKA_HEIGHTFIELDBODYDETECTOR_H