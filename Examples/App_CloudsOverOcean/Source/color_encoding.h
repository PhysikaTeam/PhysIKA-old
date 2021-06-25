#ifndef COLOR_ENCODING_H
#define COLOR_ENCODING_H

#include <map>
class MyColor
{
public:
	double blue;
	double green;
	double red;
	MyColor(double b,double g,double r)
	{
		blue =b;
		green=g;
		red=r;
	}
	MyColor(const MyColor& c)
	{
		blue=c.blue;
		green=c.green;
		red=c.red;
	}

	MyColor& operator=(const MyColor& c)
	{
		blue=c.blue;
		green=c.green;
		red=c.red;
		return *this;
	}
	MyColor()
	{

	}


};
// a struct which maps: a range of numbers[_up, _low] -> color
struct   ColorEncoder
{
	std::map<double, MyColor> color_table;
	double step;
	double scalar_low;
	MyColor col_bnd[9];
	ColorEncoder(double _up, double _low)
	{
		step = (_up-_low) / 8;
		scalar_low = _low;

	   col_bnd[0] =  MyColor(69 ,139, 0);
		col_bnd[1] = MyColor(205,186, 150);
		col_bnd[2] = MyColor(205,186, 150);
		col_bnd[3] = MyColor(139, 129, 76);
		col_bnd[4] =  MyColor(69 ,139 ,0);
		col_bnd[5] = MyColor(205,186, 150);
		col_bnd[6] = MyColor(255 ,218, 185);
		col_bnd[7] = MyColor(139, 129, 76);
		col_bnd[8] =  MyColor(255 ,218, 185);
		
	}
	
	// map to proper range
	const MyColor mapToColor(double _x)
	{
		MyColor col_to_ret, col_up, col_low;
		int low = (_x-scalar_low) / step;
		int up = low + 1;
		double w = (_x-scalar_low) / step - low;
		col_low = col_bnd[low];
		col_up = col_bnd[up];
		col_to_ret = MyColor(
				(1-w)*col_low.red + w*col_up.red,
				(1-w)*col_low.green + w*col_up.green,
				(1-w)*col_low.blue + w*col_up.blue
			);
		return col_to_ret;
	}
	
};

#endif