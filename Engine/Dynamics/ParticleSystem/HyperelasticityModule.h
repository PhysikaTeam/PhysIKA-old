#pragma once
#include "ElasticityModule.h"

namespace Physika {

	/**
	 * @brief Basis Function
	 * 
	 * @tparam T Real value
	 * @tparam n The degree of the basis
	 */
	template<typename T, int n>
	class Basis
	{
	public:
		static COMM_FUNC T A(T s) {
			return ((pow(s, n + 1) - 1) / (n + 1) + (pow(s, 1 - n) - 1) / (n - 1)) / n;
		}
		
		static COMM_FUNC T B(T s) {
			return 2 * ((pow(s, n + 1) - 1) / (n + 1) - s + 1) / n;
		}

		static COMM_FUNC T dA(T s) {
			Real sn = pow(s, n);
			return (sn - 1 / sn) / 2;
		}

		static COMM_FUNC T dB(T s) {
			return 2 * (pow(s, n) - 1);
		}
	};

	template<typename T>
	class Basis<T, 1>
	{
	public:
		static COMM_FUNC T A(T s) {
			return (s * s - 1) / 2 - log(s);
		}

		static	COMM_FUNC T B(T s) {
			return s * s - s;
		}

		static COMM_FUNC T dA(T s) {
			return s - 1 / s;
		}

		static COMM_FUNC T dB(T s) {
			return 2 * (s - 1);
		}
	};

	template<typename T>
	struct ConstantFunc
	{
		/*! Function call operator. The return value is <tt>lhs + rhs</tt>.
		*/
		COMM_FUNC inline T operator()(const T s) const { return T(1); }
	}; // end LinearFunc

	template<typename T>
	struct QuadraticFunc
	{
		COMM_FUNC inline T operator()(const T s) const {
			return Basis<T, 2>::A(s) + Basis<T, 2>::B(s);
		}
	};

	template<typename TDataType>
	class HyperelasticityModule : public ElasticityModule<TDataType>
	{
	public:
		HyperelasticityModule();
		~HyperelasticityModule() override {};
		
		enum EnergyType
		{
			Linear,
			Quadratic
		};

		void setEnergyFunction(EnergyType type) { m_energyType = type; }

	protected:
		void enforceElasticity() override;

	private:
		EnergyType m_energyType;
	};

}