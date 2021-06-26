#pragma once
enum GeometricNewType
{
	GNull,
	Discrete,
	Beam,
	Shell,
	Solid,
};

enum ElementNewType
{
	ENull,
	HLBeam,
	BSBeam,
	Spring,
	Damper,
	Truss,
	DiscreteE,
	MembraTri,
	MembraQuadR,
	BTShell,
	QPHShell,
	HLShell,
	FullIntegraShell,
	DSGShell,
	CPDSGShell,
	Tetra1st,
	Pyramid1st,
	Hexa1st,
	Hexa1stR,
	Hexa1stI,
	Penta1st,
	Penta1stR,
	SolidShell6Node,
	SolidShell8Node,
	Tetra2nd,
	Penta2nd,
	Hexa2nd,
	Tetre2ndM,
};

enum ElementSectionType
{
	EsNull,
	Discrete1st,
	Beam1st,
	Shell1st,
	Solid1st,
	Beam2nd,
	Shell2nd,
	Solid2nd,
};

enum SmoothFEMType
{
	SNull,
	ESFEM,
	CSFEM,
	NSFEM,
	SNSFEM,
};