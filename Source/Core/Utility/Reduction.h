#pragma once
namespace PhysIKA {

	template<typename T>
	class Reduction
	{
	public:
		Reduction();

		static Reduction* Create(int n);
		~Reduction();

		T accumulate(T * val, int num);

		T maximum(T* val, int num);

		T minimum(T* val, int num);

		T average(T* val, int num);

	private:
		Reduction(unsigned num);

		void allocAuxiliaryArray(int num);

		int getAuxiliaryArraySize(int n);
		
		unsigned m_num;
		
		T* m_aux;
		int m_auxNum;
	};


	template class Reduction<int>;
	template class Reduction<float>;
	template class Reduction<double>;
}
