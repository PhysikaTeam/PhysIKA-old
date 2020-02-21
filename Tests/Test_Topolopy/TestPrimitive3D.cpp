#include "gtest/gtest.h"
#include "Topology/Primitive3D.h"

using namespace PhysIKA;

TEST(Line3D, distance) {
    //arrange
    //act
    //assert
	Line3D line(Coord3D(0), Coord3D(1, 0, 0));
	Point3D point(-3, -4, 0);
    EXPECT_EQ (line.distance(point),  4);

	Line3D line2(Coord3D(0), Coord3D(0, 0, 0));
	EXPECT_EQ(line2.distance(point), 4);
}

TEST(Ray3D, distance) {
	//arrange
	//act
	//assert
	Ray3D line(Coord3D(0), Coord3D(1, 0, 0));
	Point3D point(-3, -4, 0);
	EXPECT_EQ(point.distance(line), 5);

	Ray3D line2(Coord3D(0), Coord3D(0, 0, 0));
	EXPECT_EQ(point.distance(line2), 5);
}

TEST(Segement3D, distance) {
	//arrange
	//act
	//assert
	Segment3D seg(Coord3D(0), Coord3D(1, 0, 0));
	Point3D point(-3, -4, 0);
	EXPECT_EQ(point.distance(seg), 5);

	Point3D point2(4, 4, 0);
	EXPECT_EQ(point2.distance(seg), 5);

}