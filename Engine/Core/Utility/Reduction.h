#pragma once
namespace Physika {

#define REDUCTION_BLOCK 128

	template<typename T>
	class Reduction
	{
	public:
		static Reduction* Create(int n);
		~Reduction();

		T Accumulate(T * val, int num);

		T Maximum(T* val, int num);

		T Minimum(T* val, int num);

		T Average(T* val, int num);

	private:
		Reduction(unsigned num);

		int GetAuxiliaryArraySize(int n) { return (n / REDUCTION_BLOCK + 1) + (n / (REDUCTION_BLOCK*REDUCTION_BLOCK) + REDUCTION_BLOCK); }
		
		unsigned m_num;
		
		T* m_aux;
		int m_auxNum;
	};

	template class Reduction<int>;
	template class Reduction<float>;
	template class Reduction<double>;
}
