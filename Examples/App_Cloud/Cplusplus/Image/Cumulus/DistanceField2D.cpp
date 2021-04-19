
#include "total.h"
#include "DistanceField2D.h"

DistanceField2D::DistanceField2D(void)
{
	this->x_res=PARTICLE_RES;
	this->y_res=PARTICLE_RES;
	this->x_min=-1;
	this->x_max=1;
	this->y_min=-1;
	this->y_max=1;
	this->dx=(x_max-x_min)/(x_res-1);
	this->dy=(y_max-y_min)/(y_res-1);

	this->disList=new float[x_res*y_res];
}

DistanceField2D::~DistanceField2D(void)
{
	if(disList!=NULL)
		delete [] disList;

	if( verList!=NULL)
		delete [] verList;
  
}

void DistanceField2D::vertices_get_extent( const vertex_t* vl, int np, /* in vertices */ rect_t* rc /* out extent*/ )
{ 
	if(np > 0)
	{ 
		rc->min_x = rc->max_x = vl[0].x; rc->min_y = rc->max_y = vl[0].y; 
	}
	else
	{ 
		rc->min_x = rc->min_y = rc->max_x = rc->max_y = 0; /* =0 ? no vertices at all */ 
	} 
	for(int i=1; i<np;i++)
	{ 
		if(vl[i].x < rc->min_x) 
			rc->min_x = vl[i].x; 
		if(vl[i].y < rc->min_y) 
			rc->min_y = vl[i].y; 
		if(vl[i].x > rc->max_x) 
			rc->max_x = vl[i].x; 
		if(vl[i].y > rc->max_y)
			rc->max_y = vl[i].y; 
	} 
}

int DistanceField2D::is_same( const vertex_t* l_start, const vertex_t* l_end, /* line l */ const vertex_t* p, const vertex_t* q )
{
	float dx = l_end->x - l_start->x; 
	float dy = l_end->y - l_start->y; 
	float dx1= p->x - l_start->x; 
	float dy1= p->y - l_start->y; 
	float dx2= q->x - l_end->x; 
	float dy2= q->y - l_end->y; 
	return ((dx*dy1-dy*dx1)*(dx*dy2-dy*dx2) > 0? 1 : 0); 

}

int DistanceField2D::is_intersect( const vertex_t* s1_start, const vertex_t* s1_end, const vertex_t* s2_start, const vertex_t* s2_end )
{
	return (is_same(s1_start, s1_end, s2_start, s2_end)==0 && is_same(s2_start, s2_end, s1_start, s1_end)==0)? 1: 0; 
}

int DistanceField2D::pt_in_poly( const vertex_t* vl, int np, /* polygon vl with np vertices */ const vertex_t* v )
{
	int i, j, k1, k2, c; 
	rect_t rc; 
	vertex_t w; 
	if (np < 3) 
		return 0; 
	vertices_get_extent(vl, np, &rc); 
	if (v->x < rc.min_x || v->x > rc.max_x || v->y < rc.min_y || v->y > rc.max_y) 
		return 0; 
	/* Set a horizontal beam l(*v, w) from v to the ultra right */ 
	w.x = rc.max_x + 0.00001; 
	w.y = v->y; 
	c = 0; /* Intersection points counter */ 
	for(i=0; i <np;i++)
	{ 
		j = (i+1) % np; 
		if(is_intersect(vl+i, vl+j, v, &w)) 
		{ 
			c++; 
		} 
		else if(vl[i].y==w.y) 
		{ 
			k1 = (np+i-1)%np; 
			while(k1!=i && vl[k1].y==w.y) 
				k1 = (np+k1-1)%np; 
			k2 = (i+1)%np; 
			while(k2!=i && vl[k2].y==w.y) 
				k2 = (k2+1)%np; 
			if(k1 != k2 && is_same(v, &w, vl+k1, vl+k2)==0) 
				c++; 
			if(k2 <= i) 
				break; 
			i = k2; 
		} 
	} 
	return c%2; 
}

float DistanceField2D::GetPointDistance( vertex_t p1, vertex_t p2 )
{
		 return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));  
}

float DistanceField2D::GetNearestDistance( vertex_t PA, vertex_t PB, vertex_t P3 )
{	 
	float a,b,c;  
	a=GetPointDistance(PB,P3);  
	if(a<=0.00001)  
		return 0.0f;  
	b=GetPointDistance(PA,P3);  
	if(b<=0.00001)  
		return 0.0f;  
	c=GetPointDistance(PA,PB);  
	if(c<=0.00001)  
		return a;//���PA��PB������ͬ�����˳������������ؾ���   
	//------------------------------   

	if(a*a>=b*b+c*c)
		return b;      //����Ƕ۽Ƿ���b   
	if(b*b>=a*a+c*c)
		return a;      //����Ƕ۽Ƿ���a   

	float l=(a+b+c)/2;     //�ܳ���һ��   
	float s=sqrt(l*(l-a)*(l-b)*(l-c));  //���׹�ʽ�������Ҳ������ʸ����   
	return 2*s/c;  

}

//��������Ǹ���ģ�
void DistanceField2D::CreateDisList()
{
	float max=-9999;
	for(int i=0;i<y_res;i++)
		for(int j=0;j<x_res;j++)
		{
			vertex_t vert;
			vert.x=x_min+dx*j;
			vert.y=y_min+dy*i;

			int isIn=pt_in_poly(verList,nVer,&vert);

			float  mindis=9999;
	        float tmpDis;
			for(int k=0;k<nVer;k++)
			{
				tmpDis=GetNearestDistance(verList[k],verList[(k+1)%nVer],vert);
				if(mindis>tmpDis)
					mindis=tmpDis;

			}
			if(isIn)
				mindis=-mindis;

			if(fabs(mindis)<0.0000001)
			{
				mindis=0;
			}

			disList[i*x_res+j]=mindis;
		
	
			if(fabs(mindis)>max)
				max=fabs(mindis);

		}

		//normalize the distance field

		for(int i=0;i<y_res;i++)
			for(int j=0;j<x_res;j++)
			{
	             disList[i*x_res+j]/=max;
			}

}

void DistanceField2D::CreateVerList( float* ptList, int verNumber )
{
	verList=new vertex_t[verNumber];
	for(int i=0;i<verNumber;i++)
	{
		verList[i].x=ptList[2*i+0];
		verList[i].y=ptList[2*i+1];

	}
	this->nVer=verNumber;

}

/*
void DistanceField2D::DrawDistance()
{

	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	for(int i=0;i<y_res;i++)
		for(int j=0;j<x_res;j++)
		{
			float x=x_min+dx*j;
			float y=y_min+dy*i;
			float value=disList[i*x_res+j];
            if(value<=0)
			{
				glColor3d(0,-value,0.5);
			    glVertex3d(x,y,0);

			}
		}
	 glEnd();

}
*/
float DistanceField2D::InterPolate( float x,float y )
{
	int grid_x=int((x-x_min)/dx);
	int grid_y=int((y-y_min) /dy);

	float dlta_x=(x-x_min)/dx-grid_x;
	float dlta_y=(y-y_min) /dy-grid_y;

	float  temp1=(1-dlta_x)*disList[grid_y*x_res+grid_x]+dlta_x*disList[grid_y*x_res+grid_x+1];
	float  temp2=(1-dlta_x)*disList[(grid_y+1)*x_res+grid_x]+dlta_x*disList[(grid_y+1)*x_res+grid_x+1];
	return    temp1*(1-dlta_y)+temp2*dlta_y;	
}

float DistanceField2D::GetDistance( int idx,int idy )
{

	if(idx<0||idx>x_res-1||idy<0||idy>y_res-1)
		return MAXVAL;
		
	return  disList[idy*x_res+idx];

}

Vector2  DistanceField2D::GetPos( int idx,int idy )
{
	float x=x_min+dx*idx;
	float y=y_min+dy*idy;

	return Vector2(x,y);

}
