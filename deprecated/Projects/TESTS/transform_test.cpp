#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Transform/transform_3d.h"
#include "Physika_Core/Transform/transform_2d.h"
#include "Physika_Core/Utilities/math_utilities.h"
using namespace std;
using Physika::MatrixBase;
using Physika::Transform;
using Physika::PI;
using Physika::Vector;

int main()
{
	Transform<float, 2> t(PI/2);
	cout<<t.rotateAngle()<<endl;
	cout<<PI<<endl;
	cout<<t.rotation2x2Matrix()<<endl;
	Vector<float ,2> a(0,1);
	cout<<t.rotate(a)<<endl;
	t.setScale(Vector<float, 2>(10,3));
	cout<<t.scaling(a)<<endl;
	t.setTranslation(Vector<float, 2>(5,5));
	cout<<t.transform(a)<<endl;
	t.setIdentity();
	cout<<t.transform(a)<<endl;
	int x;
	cin>>x;
	return 0;
}