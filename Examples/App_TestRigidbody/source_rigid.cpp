#include "rigid_body_demo.h"


#include "Dynamics/RigidBody/RigidUtil.h"


template<typename T> 
void out(const Quaternion<T>& q)
{
	std::cout << q.x() << "  " << q.y() << "  " << q.z() << "  " << q.w() << std::endl;
}


void out(const Vector3f& v)
{
	std::cout << v[0] << "  " << v[1] << "  " << v[2] << std::endl;
}

void out(const MatrixMN<float>& m)
{
	for (int i = 0; i < m.rows(); ++i)
	{
		for (int j = 0; j < m.cols(); ++j)
		{
			std::cout << m(i, j) << "  ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void testMatInverse()
{
	std::default_random_engine e(time(0));
	std::uniform_real_distribution<float> u(-2.0, 2.0);

	int n = 10;
	for (int casei = 0; casei < n; ++casei)
	{
		int dim = 6;
		MatrixMN<float> mat(dim, dim);
		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				mat(i, j) = u(e);
			}
		}

		MatrixMN<float> inv = RigidUtil::inverse(mat, dim);

		MatrixMN<float> iden = inv * mat;

		out(iden);
	}

}



int main()
{

	//testMatInverse();

	//demoLoadFile();
	
	//demo_PrismaticJoint();
	//demo_PlanarJoint();

	//demo_middleAxis();

	//demo_SphericalJoint();

	demo_MultiRigid<50>();

	//Quaternion<float> qx(Vector3f(1, 0, 0), 1.57);
	//Quaternion<float> qy(Vector3f(0, 1, 0), 1.57);
	//Quaternion<float> qxy = qx * qy;
	//Quaternion<float> qyx = qy * qx;

	//Vector3f tmpv(1, 2, 3);

	//out(qx.rotate(tmpv));
	//out(qy.rotate(tmpv));

	//out(qx.rotate(qy.rotate(tmpv)));
	//out(qy.rotate(qx.rotate(tmpv)));

	//out(qxy.rotate(tmpv));
	//out(qyx.rotate(tmpv));
	
	



	system("pause");
	return 0;

}

