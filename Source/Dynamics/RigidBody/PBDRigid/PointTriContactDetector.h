//#pragma once
//
//#ifndef PHYSIKA_POINTTRICONTACTDETECTION_H
//#define PHYSIKA_POINTTRICONTACTDETECTION_H
//
//#include "Framework/Framework/Node.h"
//#include "Framework/Topology/NeighborTriangleQuery.h"
//#include "Dynamics/RigidBody/ContactInfo.h"
//#include "Dynamics/RigidBody/RigidBody2.h"
//
//#include "Core/Array/DynamicArray.h"
//
//#include "Framework/Framework/CollisionModel.h"
//#include "Framework/Collision/CollisionDetectionBroadPhase.h"
//
//#include <vector>
//#include <memory>
//
//namespace PhysIKA
//{
//
//    /**
//    * @brief Contact detection of points and rigid triangle mesh.
//    */
//    class PointTriContactDetector
//    {
//
//    public:
//        PointTriContactDetector();
//
//        ~PointTriContactDetector();
//
//        void initialize();
//        //void detectAll();
//
//        //void setRigids(std::vector<RigidBody2_ptr>* prigids) { m_pRigids = prigids; }
//
//        int doBroadPhaseDetect(Real dt);
//        int doNarrowDetect(Real dt);
//
//        int contectDetection(DeviceDArray<ContactInfo<double>>& contacts,  int i, int j, Real dt=0.02, int begin=0);
//        int detectPointTriCondidate(DeviceDArray<PointTriContact<float>>& contacts, int i, int j, Real dt = 0.02, int begin = 0);
//
//        int detectFromPointTriCondidate(DeviceDArray<ContactInfo<double>>& contacts);
//
//        void updateTriPoint();
//
//        void initRigidRadius();
//
//        DeviceDArray<ContactInfo<double>>& getContacts() { return m_contactInfos; }
//        const DeviceDArray<ContactInfo<double>>& getContacts()const { return m_contactInfos; }
//
//        void setMaxContactPerPair(float mc) { m_maxContactPerPair = mc; }
//        void setMaxContactPairs(float mp) { m_maxContactPair = mp; }
//        void setDetectionExtention(float ext) { m_detectionExt = ext; }
//    private:
//        void _updateRigidRotInfo();
//        void _updateRigidPosInfo();
//
//    protected:
//
//
//
//        std::vector<ArrayField<Vector3f, DeviceType::GPU>> m_localMeshPoints;
//        //std::vector<ArrayField<Vector3f, DeviceType::GPU>> m_rigidPoints;
//
//        ArrayField<Vector3f, DeviceType::GPU> m_localMeshPoints;    // save all mesh in a single array.
//        ArrayField<Vector3f, DeviceType::GPU> m_globalMeshPoints;
//        ArrayField< TopologyModule::Triangle, DeviceType::GPU> m_triVerIndices;
//
//        ArrayField<int, DeviceType::GPU> m_meshBelongTo;
//        ArrayField<int, DeviceType::GPU> m_pointIndexOffset;
//
//        // Point position, should be update before every detection.
//        ArrayField<Vector3f, DeviceType::GPU> m_inPoints;
//
//        std::shared_ptr<NeighborTriangleQuery<DataType3f>> m_triQuery;
//
//        NeighborField<int> m_neighborhood;
//
//        DeviceArray<int> m_counter;
//
//        //std::shared_ptr<CollisionDetectionBroadPhase<DataType3f>> m_broadPhaseCD;
//        //NeighborField<int> m_broadNeighbor;
//        //DeviceArray<float> m_rigidRadius;
//        //DeviceArray<AABB> m_rigidAABB;
//        DeviceArray<Vector3f> m_triPos;
//        HostArray<Vector3f> m_triPosHost;
//
//        DeviceArray<Quaternionf> m_triRot;
//        HostArray<Quaternionf> m_triRotHost;
//
//
//        DeviceDArray<ContactPair> m_contactPairs;
//        HostDArray<ContactPair> m_contactPairsHost;
//
//        DeviceDArray<ContactInfo<double>> m_contactInfos;
//        std::shared_ptr<Reduction<float>> m_reductionf;
//
//
//        DeviceDArray<PointTriContact<float>> m_pointTriCondidate;
//
//        int m_maxContactPerPair = 100;
//        int m_maxContactPair = 1000;
//
//        float m_detectionExt = 0.05;
//
//    };
//
//
//
//}
//
//
//
//#endif // PHYSIKA_POINTTRICONTACTDETECTION_H