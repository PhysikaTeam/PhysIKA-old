#ifndef __SIMULATIONMODEL_H__
#define __SIMULATIONMODEL_H__

#include "Common/Common.h"
#include <vector>
#include "Demos/Simulation/RigidBody.h"
#include "Demos/Simulation/ParticleData.h"
#include "TriangleModel.h"
#include "TetModel.h"
#include "LineModel.h"
#include "Demos/Utils/ObjectArray.h"

namespace PBD 
{	
	class Constraint;

	class SimulationModel
	{
		public:
			SimulationModel();
			virtual ~SimulationModel();

			typedef std::vector<Constraint*> ConstraintVector;
			typedef Utilities::ObjectArray<RigidBodyContactConstraint> RigidBodyContactConstraintVector;
			typedef Utilities::ObjectArray<ParticleRigidBodyContactConstraint> ParticleRigidBodyContactConstraintVector;
			typedef std::vector<RigidBody*> RigidBodyVector;
			typedef std::vector<TriangleModel*> TriangleModelVector;
			typedef std::vector<TetModel*> TetModelVector;
			typedef std::vector<LineModel*> LineModelVector;
			typedef std::vector<unsigned int> ConstraintGroup;
			typedef std::vector<ConstraintGroup> ConstraintGroupVector;


		protected:
			RigidBodyVector m_rigidBodies;
			TriangleModelVector m_triangleModels;
			TetModelVector m_tetModels;
			LineModelVector m_lineModels;
			ParticleData m_particles;
			OrientationData m_orientations;
			ConstraintVector m_constraints;
			RigidBodyContactConstraintVector m_rigidBodyContactConstraints;
			ParticleRigidBodyContactConstraintVector m_particleRigidBodyContactConstraints;
			ConstraintGroupVector m_constraintGroups;

			Real m_cloth_stiffness;
			Real m_cloth_bendingStiffness;
			Real m_cloth_xxStiffness;
			Real m_cloth_yyStiffness;
			Real m_cloth_xyStiffness;
			Real m_cloth_xyPoissonRatio;
			Real m_cloth_yxPoissonRatio;
			bool  m_cloth_normalizeStretch;
			bool  m_cloth_normalizeShear;

			Real m_solid_stiffness;
			Real m_solid_poissonRatio;
			bool m_solid_normalizeStretch;
			bool m_solid_normalizeShear;

			Real m_contactStiffnessRigidBody;
			Real m_contactStiffnessParticleRigidBody;

			Real m_rod_stretchingStiffness;
			Real m_rod_shearingStiffness1;
			Real m_rod_shearingStiffness2;
			Real m_rod_bendingStiffness1;
			Real m_rod_bendingStiffness2;
			Real m_rod_twistingStiffness;

	public:
			void reset();			
			void cleanup();

			RigidBodyVector &getRigidBodies();
			ParticleData &getParticles();
			OrientationData &getOrientations();
			TriangleModelVector &getTriangleModels();
			TetModelVector &getTetModels();
			LineModelVector &getLineModels();
			ConstraintVector &getConstraints();
			RigidBodyContactConstraintVector &getRigidBodyContactConstraints();
			ParticleRigidBodyContactConstraintVector &getParticleRigidBodyContactConstraints();
			ConstraintGroupVector &getConstraintGroups();
			bool m_groupsInitialized;

			void resetContacts();

			void addTriangleModel(
				const unsigned int nPoints,
				const unsigned int nFaces,
				Vector3r *points,
				unsigned int* indices,
				const TriangleModel::ParticleMesh::UVIndices& uvIndices,
				const TriangleModel::ParticleMesh::UVs& uvs);

			void addTetModel(
				const unsigned int nPoints, 
				const unsigned int nTets, 
				Vector3r *points,
				unsigned int* indices);

			void addLineModel(
				const unsigned int nPoints,
				const unsigned int nQuaternions,
				Vector3r *points,
				Quaternionr *quaternions,
				unsigned int *indices,
				unsigned int *indicesQuaternions);

			void updateConstraints();
			void initConstraintGroups();

			bool addBallJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos);
			bool addBallOnLineJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &dir);
			bool addHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addTargetAngleMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addTargetVelocityMotorHingeJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addUniversalJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis1, const Vector3r &axis2);
			bool addSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addTargetPositionMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addTargetVelocityMotorSliderJoint(const unsigned int rbIndex1, const unsigned int rbIndex2, const Vector3r &pos, const Vector3r &axis);
			bool addRigidBodyParticleBallJoint(const unsigned int rbIndex, const unsigned int particleIndex);
			bool addRigidBodyContactConstraint(const unsigned int rbIndex1, const unsigned int rbIndex2, 
					const Vector3r &cp1, const Vector3r &cp2,	
					const Vector3r &normal, const Real dist, 
					const Real restitutionCoeff, const Real frictionCoeff);
			bool addParticleRigidBodyContactConstraint(const unsigned int particleIndex, const unsigned int rbIndex, 
					const Vector3r &cp1, const Vector3r &cp2, 
					const Vector3r &normal, const Real dist,
					const Real restitutionCoeff, const Real frictionCoeff);

			bool addDistanceConstraint(const unsigned int particle1, const unsigned int particle2);
			bool addDihedralConstraint(	const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4);
			bool addIsometricBendingConstraint(const unsigned int particle1, const unsigned int particle2,
										const unsigned int particle3, const unsigned int particle4);
			bool addFEMTriangleConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int particle3);
			bool addStrainTriangleConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int particle3);
			bool addVolumeConstraint(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4);
			bool addFEMTetConstraint(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4);
			bool addStrainTetConstraint(const unsigned int particle1, const unsigned int particle2,
									const unsigned int particle3, const unsigned int particle4);
			bool addShapeMatchingConstraint(const unsigned int numberOfParticles, const unsigned int particleIndices[], const unsigned int numClusters[]);
			bool addStretchShearConstraint(const unsigned int particle1, const unsigned int particle2, const unsigned int quaternion1);
			bool addBendTwistConstraint(const unsigned int quaternion1, const unsigned int quaternion2);

			Real getClothStiffness() const { return m_cloth_stiffness; }
			void setClothStiffness(Real val) { m_cloth_stiffness = val; }
			Real getClothBendingStiffness() const { return m_cloth_bendingStiffness; }
			void setClothBendingStiffness(Real val) { m_cloth_bendingStiffness = val; }
			Real getClothXXStiffness() const { return m_cloth_xxStiffness; }
			void setClothXXStiffness(Real val) { m_cloth_xxStiffness = val; }
			Real getClothYYStiffness() const { return m_cloth_yyStiffness; }
			void setClothYYStiffness(Real val) { m_cloth_yyStiffness = val; }
			Real getClothXYStiffness() const { return m_cloth_xyStiffness; }
			void setClothXYStiffness(Real val) { m_cloth_xyStiffness = val; }
			bool getClothNormalizeStretch() const { return m_cloth_normalizeStretch; }
			void setClothNormalizeStretch(bool val) { m_cloth_normalizeStretch = val; }
			bool getClothNormalizeShear() const { return m_cloth_normalizeShear; }
			void setClothNormalizeShear(bool val) { m_cloth_normalizeShear = val; }
			Real getClothXYPoissonRatio() const { return m_cloth_xyPoissonRatio; }
			void setClothXYPoissonRatio(Real val) { m_cloth_xyPoissonRatio = val; }
			Real getClothYXPoissonRatio() const { return m_cloth_yxPoissonRatio; }
			void setClothYXPoissonRatio(Real val) { m_cloth_yxPoissonRatio = val; }

			Real getSolidStiffness() const { return m_solid_stiffness; }
			void setSolidStiffness(Real val) { m_solid_stiffness = val; }
			Real getSolidPoissonRatio() { return m_solid_poissonRatio; }
			void setSolidPoissonRatio(const Real val) { m_solid_poissonRatio = val; }
			bool getSolidNormalizeStretch() const { return m_solid_normalizeStretch; }
			void setSolidNormalizeStretch(bool val) { m_solid_normalizeStretch = val; }
			bool getSolidNormalizeShear() const { return m_solid_normalizeShear; }
			void setSolidNormalizeShear(bool val) { m_solid_normalizeShear = val; }

			Real getContactStiffnessRigidBody() const { return m_contactStiffnessRigidBody; }
			void setContactStiffnessRigidBody(Real val) { m_contactStiffnessRigidBody = val; }
			Real getContactStiffnessParticleRigidBody() const { return m_contactStiffnessParticleRigidBody; }
			void setContactStiffnessParticleRigidBody(Real val) { m_contactStiffnessParticleRigidBody = val; }
		
		    Real getRodStretchingStiffness() const { return m_rod_stretchingStiffness;  }
			void setRodStretchingStiffness(Real val) { m_rod_stretchingStiffness = val; }
			Real getRodShearingStiffness1() const { return m_rod_shearingStiffness1; }
			void setRodShearingStiffness1(Real val) { m_rod_shearingStiffness1 = val; }
			Real getRodShearingStiffness2() const { return m_rod_shearingStiffness2; }
			void setRodShearingStiffness2(Real val) { m_rod_shearingStiffness2 = val; }
			Real getRodBendingStiffness1() const { return m_rod_bendingStiffness1; }
			void setRodBendingStiffness1(Real val) { m_rod_bendingStiffness1 = val; }
			Real getRodBendingStiffness2() const { return m_rod_bendingStiffness2; }
			void setRodBendingStiffness2(Real val) { m_rod_bendingStiffness2 = val; }
			Real getRodTwistingStiffness() const { return m_rod_twistingStiffness; }
			void setRodTwistingStiffness(Real val) { m_rod_twistingStiffness = val; }
	};
}

#endif