#include "Dynamics/RigidBody/SystemMotionState.h"
#include "RigidBodyRoot.h"

namespace PhysIKA
{
	
	template<typename T>
	void _addDvec(std::vector<T>& v, const std::vector<T>& dv, double dt)
	{
		int n = v.size();
		for (int i = 0; i < n; ++i)
		{
			v[i] = v[i] + dv[i] * dt;
		}
	}

	template<typename T>
	void _addDvec(Vectornd<T>& v, const Vectornd<T>& dv, double dt)
	{
		int n = v.size();
		for (int i = 0; i < n; ++i)
		{
			v[i] += dv[i] * dt;
		}
	}

	
	SystemMotionState & SystemMotionState::addDs(const DSystemMotionState & ds, double dt)
	{
		_addDvec(this->m_rel_r, ds.m_rel_r, dt);
		_addDvec(this->m_rel_q, ds.m_rel_q, dt);
		_addDvec(this->m_v, ds.m_v, dt);
		_addDvec(this->m_dq, ds.m_dq, dt);


		// /
		if (this->m_root)
		{
			auto all_node_pairs = static_cast<RigidBodyRoot<DataType3f>*>(this->m_root)->getAllParentidNodePair();

			for (int i = 0; i < all_node_pairs.size(); ++i)
			{
				RigidBody2_ptr cur_node = all_node_pairs[i].second;
				int parent_id = all_node_pairs[i].first;
				
				this->m_X[i] = Transform3d<float>(this->m_rel_r[i], this->m_rel_q[i].getConjugate());


				if (parent_id >= 0)
				{
					this->m_global_r[i] = this->m_global_r[parent_id] + this->m_global_q[parent_id].rotate(this->m_rel_r[i]);
					this->m_global_q[i] = this->m_global_q[parent_id] * this->m_rel_q[i];
				}
				else
				{
					this->m_global_r[i] = this->m_rel_r[i];
					this->m_global_q[i] = this->m_rel_q[i];
				}
				this->m_global_q[i].normalize();
			}
		}

		return *this;
	}

	void SystemMotionState::updateGlobalInfo()
	{
		RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*> (m_root);

		auto nodePairs = root->getAllParentidNodePair();

		for (int i = 0; i < nodePairs.size(); ++i)
		{
			RigidBody2_ptr cur_node = nodePairs[i].second;
			int cur_id = cur_node->getId();
			int parent_id = nodePairs[i].first;

			this->m_X[cur_id].set(this->m_rel_r[cur_id], this->m_rel_q[cur_id].getConjugate());

			if (parent_id >= 0)
			{
				m_global_r[cur_id] = m_global_r[parent_id] + m_global_q[parent_id].rotate(m_rel_r[cur_id]);
				m_global_q[cur_id] = m_global_q[parent_id] * m_rel_q[cur_id];
			}
			else
			{
				m_global_r[cur_id] = m_rel_r[cur_id];
				m_global_q[cur_id] = m_rel_q[cur_id];
			}
		}
	}

}