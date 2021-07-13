#include "urdf.h"

#include "tinyxml/tinyxml2.h"
#include "Dynamics/RigidBody/RevoluteJoint.h"
#include "Core/Quaternion/quaternion.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "Dynamics/RigidBody/FixedJoint.h"

#include "SystemState.h"

#include "Transform3d.h"

#include <queue>
#include <map>
#include <string>
#include <sstream>
#include <vector>

//using namespace tinyxml2;

namespace PhysIKA {

RigidBodyRoot_ptr PhysIKA::Urdf::loadFile(std::string filename)
{
    // ------  load urdf file  -------

    // open file
    tinyxml2::XMLDocument doc;
    int                   loadState = doc.LoadFile(filename.c_str());
    tinyxml2::XMLElement* urdf_root = doc.RootElement();

    RigidBodyRoot_ptr                                     rigid_root = std::make_shared<RigidBodyRoot<DataType3f>>(urdf_root->Attribute("name"));
    std::map<std::string, RigidBody2_ptr>                 name_rigid_pairs;
    std::map<std::string, std::shared_ptr<RevoluteJoint>> name_joint_pairs;
    std::map<Node*, int>                                  rigid_idx_pairs;
    std::map<Joint*, int>                                 joint_idx_pairs;

    //// sysytem state and system motion state
    //std::shared_ptr<SystemMotionState> motion_state = std::make_shared<SystemMotionState>(rigid_root.get());
    //std::shared_ptr<SystemState> system_state = std::make_shared<SystemState>();
    //system_state->m_motionState = motion_state;
    //rigid_root->setSystemState(system_state);

    std::shared_ptr<SystemState>       system_state = rigid_root->getSystemState();
    std::shared_ptr<SystemMotionState> motion_state = system_state->m_motionState;

    // read xml elements, and save them into maps
    std::vector<RigidBody2_ptr>    all_node;
    std::vector<Vector3f>          all_r_r2j;
    std::vector<Quaternion<float>> all_q_r2j;
    std::vector<Vector3f>          all_r_j2r;
    std::vector<Quaternion<float>> all_q_j2r;
    std::vector<Vector3f>          all_joint_axis;
    {
        // find all link elements
        tinyxml2::XMLElement* urdf_link = urdf_root->FirstChildElement("link");
        while (urdf_link)
        {
            // generate a new rigid body
            RigidBody2_ptr cur_rigid                       = std::make_shared<RigidBody2<DataType3f>>(urdf_link->Attribute("name"));
            name_rigid_pairs[urdf_link->Attribute("name")] = cur_rigid;
            all_node.push_back(cur_rigid);
            rigid_idx_pairs[cur_rigid.get()] = all_node.size() - 1;

            // -- link inertial --
            auto link_inertial = urdf_link->FirstChildElement("inertial");
            {
                // relative position info
                auto              inertial_origin = link_inertial->FirstChildElement("origin");
                std::stringstream ss_origin_xyz(inertial_origin->Attribute("xyz"));
                std::stringstream ss_origin_rpy(inertial_origin->Attribute("rpy"));
                Vector3f          cur_r;
                Vector3f          cur_w;
                ss_origin_rpy >> cur_w[1] >> cur_w[2] >> cur_w[0];
                ss_origin_xyz >> cur_r[0] >> cur_r[1] >> cur_r[2];
                Quaternion<float> q_rotate;
                q_rotate.set(cur_w);
                all_r_j2r.push_back(cur_r);
                all_q_j2r.push_back(q_rotate);

                // rigid body property info. inertia & mass
                auto              inertial_mass = link_inertial->FirstChildElement("mass");
                std::stringstream ss_mass_value(inertial_mass->Attribute("value"));
                double            cur_mass;
                ss_mass_value >> cur_mass;

                auto              inertial_inertia = link_inertial->FirstChildElement("inertia");
                std::stringstream ss_inertia_ixx(inertial_inertia->Attribute("ixx"));
                std::stringstream ss_inertia_ixy(inertial_inertia->Attribute("ixy"));
                std::stringstream ss_inertia_ixz(inertial_inertia->Attribute("ixz"));
                std::stringstream ss_inertia_iyy(inertial_inertia->Attribute("iyy"));
                std::stringstream ss_inertia_iyz(inertial_inertia->Attribute("iyz"));
                std::stringstream ss_inertia_izz(inertial_inertia->Attribute("izz"));
                double            ixx, ixy, ixz, iyy, iyz, izz;
                ss_inertia_ixx >> ixx;
                ss_inertia_ixy >> ixy;
                ss_inertia_ixz >> ixz;
                ss_inertia_iyy >> iyy;
                ss_inertia_iyz >> iyz;
                ss_inertia_izz >> izz;
                Inertia<float> cur_i;
                //cur_i(0, 0) = ixx;    cur_i(0, 1) = ixy;    cur_i(0, 2) = ixz;
                //cur_i(1, 0) = ixy;    cur_i(1, 1) = iyy;    cur_i(1, 2) = iyz;
                //cur_i(2, 0) = ixz;    cur_i(2, 1) = iyz;    cur_i(2, 2) = izz;
                //cur_i(3, 3) = cur_mass;
                //cur_i(4, 4) = cur_mass;
                //cur_i(5, 5) = cur_mass;
                cur_i.setInertiaDiagonal(Vector3f(ixx, iyy, izz));
                cur_i.setMass(cur_mass);
                cur_rigid->setI(cur_i);
            }

            // rigid geometry info
            auto link_visual = urdf_link->FirstChildElement("visual");
            if (link_visual)
            {
                auto visual_geometry = link_visual->FirstChildElement("geometry");
                if (visual_geometry)
                {
                    auto geometry_box = visual_geometry->FirstChildElement("box");
                    if (geometry_box)
                    {
                        std::stringstream sssize(geometry_box->Attribute("size"));
                        float             sx, sy, sz;
                        sssize >> sx >> sy >> sz;
                        cur_rigid->setGeometrySize(sx, sy, sz);
                    }
                }
            }

            // next link
            urdf_link = urdf_link->NextSiblingElement("link");
        }
        rigid_idx_pairs[rigid_root.get()] = -1;

        // find all joint elements
        tinyxml2::XMLElement* urdf_joint = urdf_root->FirstChildElement("joint");
        while (urdf_joint)
        {
            // generate a new joint
            std::shared_ptr<RevoluteJoint> cur_joint        = std::make_shared<RevoluteJoint>(urdf_joint->Attribute("name"));
            name_joint_pairs[urdf_joint->Attribute("name")] = cur_joint;
            joint_idx_pairs[cur_joint.get()]                = name_joint_pairs.size() - 1;

            // find parent and child
            RigidBody2_ptr joint_parent = name_rigid_pairs[urdf_joint->FirstChildElement("parent")->Attribute("link")];
            RigidBody2_ptr joint_child  = name_rigid_pairs[urdf_joint->FirstChildElement("child")->Attribute("link")];
            cur_joint->setRigidBody(joint_parent.get(), joint_child.get());
            joint_parent->addChildJoint(cur_joint);
            joint_parent->addChild(joint_child);
            joint_child->setParentJoint(cur_joint.get());

            // find joint position info
            auto              joint_origin = urdf_joint->FirstChildElement("origin");
            std::string       xyz          = joint_origin->Attribute("xyz");
            std::string       rpy          = joint_origin->Attribute("rpy");
            Vector3f          cur_r;
            Quaternion<float> q;
            double or, op, oy;
            std::stringstream ssxyz(xyz);
            std::stringstream ssrpy(rpy);
            ssxyz >> cur_r[0] >> cur_r[1] >> cur_r[2];
            ssrpy >> or >> op >> oy;
            q.set(Vector3f(oy, or, op));
            all_r_r2j.push_back(cur_r);
            all_q_r2j.push_back(q);

            // joint property info
            // axis is expressed in child node frame.
            auto              joint_axis = urdf_joint->FirstChildElement("axis");
            std::string       axis_xyz   = joint_axis->Attribute("xyz");
            Vector3f          axis_value;
            std::stringstream ssaxis(axis_xyz);
            ssaxis >> axis_value[0] >> axis_value[1] >> axis_value[2];
            all_joint_axis.push_back(axis_value);

            // next joint
            urdf_joint = urdf_joint->NextSiblingElement("joint");
        }
    }

    // ------  build PhysIKA simulation tree  -------

    int n_node = all_node.size();  // number of rigid bodies. Root node is not included
    motion_state->setRigidNum(n_node);

    int count = 0;
    while (!name_rigid_pairs.empty())
    {
        auto rigid_iter = name_rigid_pairs.begin();

        std::shared_ptr<FixedJoint> base_joint = std::make_shared<FixedJoint>("from_base");

        base_joint->setRigidBody(rigid_root.get(), rigid_iter->second.get());
        rigid_root->addChildJoint(base_joint);
        rigid_root->addChild(rigid_iter->second);
        rigid_iter->second->setParentJoint(base_joint.get());

        joint_idx_pairs[base_joint.get()] = joint_idx_pairs.size();
        all_r_r2j.push_back(Vector3f());
        all_q_r2j.push_back(Quaternion<float>());

        // visit the rigid body tree
        std::queue<std::shared_ptr<Node>> to_be_handle;
        to_be_handle.push(rigid_iter->second);
        while (!to_be_handle.empty())
        {
            std::shared_ptr<Node> cur_rigid = to_be_handle.front();
            to_be_handle.pop();

            std::dynamic_pointer_cast<RigidBody2<DataType3f>>(cur_rigid)->setId(count++);
            name_rigid_pairs.erase(cur_rigid->getName());

            // push child rigid
            //const ListPtr<Joint> & child_joints = cur_rigid->getChildJoints();
            const ListPtr<Node>& childs = cur_rigid->getChildren();
            for (auto iter_ = childs.begin(); iter_ != childs.end(); ++iter_)
            {
                to_be_handle.push(*iter_);
            }
        }
    }

    rigid_root->updateTree();

    // ------------
    {
        std::queue<RigidBody2_ptr> to_be_handle;
        //to_be_handle.push(rigid_root);
        RigidBody2_ptr cur_rigid = rigid_root;

        //to_be_handle.push(rigid_root);
        //while (!to_be_handle.empty())
        do
        {

            const ListPtr<Node>& childs = cur_rigid->getChildren();
            for (auto iter_ = childs.begin(); iter_ != childs.end(); ++iter_)
            {
                to_be_handle.push(std::dynamic_pointer_cast<RigidBody2<DataType3f>>(*iter_));
            }

            cur_rigid = to_be_handle.front();
            to_be_handle.pop();

            // handle
            RevoluteJoint*          cur_parent_joint = static_cast<RevoluteJoint*>(cur_rigid->getParentJoint());
            RigidBody2<DataType3f>* cur_parent_node  = static_cast<RigidBody2<DataType3f>*>(cur_parent_joint->getParent());

            int cur_id    = rigid_idx_pairs[cur_rigid.get()];
            int parent_id = rigid_idx_pairs[cur_parent_node];
            int joint_id  = joint_idx_pairs[cur_parent_joint];

            // transformation from parent node frame to joint frame.
            // Conjugate here!
            Transform3d<float> X_r2j(all_r_r2j[joint_id], all_q_r2j[joint_id].getConjugate());
            cur_parent_joint->setXT(X_r2j);

            // transformation from joint frame to child node frame.
            Transform3d<float> X_j2r(all_r_j2r[cur_id], all_q_j2r[cur_id].getConjugate());

            // transformation from parent node frame to child node frame.
            motion_state->m_X[cur_id] = X_j2r * X_r2j;

            //Quaternion<float> tmp_q = X_r2j.getRotation();

            // position of child node relative to parent node. In parent frame.
            //Vector3f rotated_r = X_r2j.getRotation().getConjugate().rotate(all_r_j2r[cur_id]);
            Vector3f rotated_r            = all_q_r2j[joint_id].rotate(all_r_j2r[cur_id]);
            Vector3f rel_r                = (rotated_r + all_r_r2j[joint_id]);
            motion_state->m_rel_r[cur_id] = rel_r;

            //// ---------------------------------------------------
            //Vector3f axis;
            //float ran;
            //all_q_r2j[joint_id].getRotation(ran, axis);
            //Vector3f new_rel_r = all_q_r2j[joint_id].rotate(all_r_j2r[cur_id]);

            //Quaternion<float> tmp_q = all_q_r2j[joint_id];
            //Matrix3f tmp_m = tmp_q.get3x3Matrix();
            //Vector3f new_rel_r2 = tmp_m *(all_r_j2r[cur_id]);
            //bool is_equal = (new_rel_r2 == new_rel_r);
            //Quaternion<float> tmp_q2(tmp_m);
            //Vector3f new_rel_r3 = tmp_q2.rotate(all_r_j2r[cur_id]);
            //bool is_equal2 = (new_rel_r3 == new_rel_r);

            // rotation of child node relative to parent node.
            // Be care of the order of multiplication
            Quaternion<float> rel_q       = all_q_r2j[joint_id] * all_q_j2r[cur_id];
            motion_state->m_rel_q[cur_id] = rel_q;

            //// ---------------------
            //Vector3f axis2;
            //float ran2;
            //rel_q.getRotation(ran2, axis2);
            //
            //Vector3f axis3;
            //float ran3;
            //all_q_r2j[joint_id].getRotation(ran3, axis3);

            //Vector3f axis4;
            //float ran4;
            //all_q_j2r[cur_id].getRotation(ran4, axis4);

            if (parent_id >= 0)
            {
                // global translation
                //Transform3d<float> parent_trans(motion_state->m_global_r[parent_id],
                //    motion_state->m_global_q[parent_id].getConjugate());
                Vector3f global_r                    = motion_state->globalPosition[parent_id] + motion_state->globalRotation[parent_id].rotate(rel_r);
                motion_state->globalPosition[cur_id] = global_r;

                //Vector3f tmp_rel_r = motion_state->m_global_q[parent_id].rotate(rel_r);
                //Vector3f axis;
                //float ran;
                //motion_state->m_global_q[parent_id].getRotation(ran, axis);

                // global rotation
                Quaternion<float> global_q           = motion_state->globalRotation[parent_id] * rel_q;
                motion_state->globalRotation[cur_id] = global_q;

                // joint space matrix info
                Quaternion<float> q_j2r_conj = all_q_j2r[cur_id].getConjugate();
                cur_parent_joint->setJointInfo(q_j2r_conj.rotate(all_joint_axis[joint_id]), q_j2r_conj.rotate(-all_r_j2r[cur_id]));
            }
            else
            {
                motion_state->globalPosition[cur_id] = motion_state->m_rel_r[cur_id];
                motion_state->globalRotation[cur_id] = motion_state->m_rel_q[cur_id];
            }

            //} while (!to_be_handle.empty());
        } while (cur_rigid->getChildren().size() > 0);
    }
    return rigid_root;
}

}  // namespace PhysIKA