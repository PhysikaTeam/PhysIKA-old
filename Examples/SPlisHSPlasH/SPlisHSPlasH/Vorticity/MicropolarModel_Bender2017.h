#ifndef __MicropolarModel_Bender2017_h__
#define __MicropolarModel_Bender2017_h__

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/FluidModel.h"
#include "VorticityBase.h"

namespace SPH
{
	/** \brief This class implements the micropolar material model introduced
	* by Bender et al. [BKKW17].
	*
	* References:
	* - [BKKW17] Jan Bender, Dan Koschier, Tassilo Kugelstadt, and Marcel Weiler. A micropolar material model for turbulent SPH fluids. In ACM SIGGRAPH / Eurographics Symposium on Computer Animation, SCA '17. ACM, 2017. URL: http://doi.acm.org/10.1145/3099564.3099578
	*/
	class MicropolarModel_Bender2017 : public VorticityBase
	{
	protected:
		std::vector<Vector3r> m_angularAcceleration;
		std::vector<Vector3r> m_omega;
		Real m_viscosityOmega;
		Real m_inertiaInverse;

		virtual void initParameters();

	public:
		static int VISCOSITY_OMEGA;
		static int INERTIA_INVERSE;

		MicropolarModel_Bender2017(FluidModel *model);
		virtual ~MicropolarModel_Bender2017(void);

		virtual void step();
		virtual void reset();

		virtual void performNeighborhoodSearchSort();

		FORCE_INLINE const Vector3r& getAngularAcceleration(const unsigned int i) const
		{
			return m_angularAcceleration[i];
		}

		FORCE_INLINE Vector3r& getAngularAcceleration(const unsigned int i)
		{
			return m_angularAcceleration[i];
		}

		FORCE_INLINE void setAngularAcceleration(const unsigned int i, const Vector3r& val)
		{
			m_angularAcceleration[i] = val;
		}

		FORCE_INLINE const Vector3r& getAngularVelocity(const unsigned int i) const
		{
			return m_omega[i];
		}

		FORCE_INLINE Vector3r& getAngularVelocity(const unsigned int i)
		{
			return m_omega[i];
		}

		FORCE_INLINE void setAngularVelocity(const unsigned int i, const Vector3r& val)
		{
			m_omega[i] = val;
		}
	};
}

#endif
