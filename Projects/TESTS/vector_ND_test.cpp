#include<iostream>
#include"Physika_Core\Vectors\vector_Nd.h"
#include"Physika_Core\Timer\timer.h"
#define MAX 10000
using Physika::VectorND;
using Physika::Timer;
using namespace std;

int main()
{
	int i;
	float temp = 1.00000001;
	VectorND<float> a(6),b(6);
	a[0] = 1.0, a[1] = 1.0, a[2] = a[3] = 1.000001, a[4] = a[5] = 1.0000001;
	b = a;
	Timer time;
	time.startTimer();
	for (i = 0; i < MAX; ++i)
	{
		b=b*b.dot(a);
	}
	time.stopTimer();
	cout << MAX << " times operations of b=b.dot(a) costs time:" << time.getElapsedTime() << endl;
	time.startTimer();
	for (i = 0; i < MAX; ++i)
	{
		temp = temp*temp; temp = temp*temp; temp = temp*temp;
		temp = temp*temp; temp = temp*temp; temp = temp*temp;
		temp = temp*temp; temp = temp*temp; temp = temp*temp;
		temp = temp*temp; temp = temp*temp; temp = temp*temp;     //12 times float multiplications
		temp = temp + temp; temp = temp + temp; temp = temp + temp; temp = temp + temp; temp = temp + temp; //5 times additions
	}
	time.stopTimer();
	cout << "corespodent theorical float operations costs time:" << time.getElapsedTime() << endl;
	cout << b[5] << endl;
	cout << temp << endl;
	return 0;
}