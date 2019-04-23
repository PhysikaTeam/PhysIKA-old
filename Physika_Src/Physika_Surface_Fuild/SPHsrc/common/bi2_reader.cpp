#include "bi2_reader.h"

BI2Reader::BI2Reader(string dir)
{
	BI2FilePnt = fopen(dir.c_str(),"rb");
	if (BI2FilePnt == NULL)
	{
		printf("cannot open the file");
	}
}
BI2Reader::BI2Reader(char* dir)
{
	BI2FilePnt = fopen(dir,"rb");
	if (BI2FilePnt == NULL)
	{
		printf("cannot open the file");
	}
}
void BI2Reader::GetInfo(bool fileout)
{
	memset(&info,0,sizeof(StInfoFileBi2));
	StHeadFmtBin hfmt;
	fread((char*)&hfmt,sizeof(StHeadFmtBin),1,BI2FilePnt);
	printf("%d\n",hfmt.bitorder);
	bool rendi;
	if(!strcmp(hfmt.titu,"#File BINX-020 ")||(fileout&&hfmt.full==2)||(!fileout&&hfmt.full<=1))
	{
		info.fmt=20;
		info.bitorder=hfmt.bitorder;
		info.full=(hfmt.full!=0);
		info.data2d=(hfmt.data2d!=0);
		rendi=(hfmt.bitorder!=byte(fun::GetByteOrder()));
		if(hfmt.full){
			StHeadDatFullBi2 hdat;
			fread((char*)&hdat,sizeof(StHeadDatFullBi2),1,BI2FilePnt);
			if(rendi) fun::ReverseByteOrder((int*)&hdat,sizeof(StHeadDatFullBi2)/4);//-Conversion Big/LittleEndian
			info.dp=hdat.dp;
			info.h=hdat.h;
			info.b=hdat.b;
			info.rhop0=hdat.rhop0;
			info.gamma=hdat.gamma;
			info.massbound=hdat.massbound;
			info.massfluid=hdat.massfluid;
			info.np=hdat.np;
			info.nfixed=hdat.nfixed;
			info.nmoving=hdat.nmoving;
			info.nfloat=hdat.nfloat;
			info.nfluidout=hdat.nfluidout;
			info.time=hdat.timestep;
		}
		else{
			StHeadDatBi2 hd;
			fread((char*)&hd,sizeof(StHeadDatBi2),1,BI2FilePnt);
			if(rendi)fun::ReverseByteOrder((int*)&hd,sizeof(StHeadDatBi2)/4);//-Conversion Big/LittleEndian
			info.np=hd.np;
			info.nfixed=hd.nfixed;
			info.nmoving=hd.nmoving;
			info.nfloat=hd.nfloat;
			info.nfluidout=hd.nfluidout;
			info.time=hd.timestep;
		}
	}

	info.nbound=info.nfixed+info.nmoving+info.nfloat;
	info.nfluid=info.np-info.nbound;
	info.Id=new unsigned[info.np];        
	info.Pos=new tfloat3[info.np];       
	info.Vel=new tfloat3[info.np];  
	info.Rhop=new float[info.np];       
	info.OrderOut=new int[info.nfluid];
	for(unsigned c=0;c<info.np;c++) info.Id[c]=c;
	for(unsigned c=0;c<info.nfluid;c++) info.OrderOut[c]=-1;
	info.NFluidOut=0;
	info.OutCountSaved=0;
	info.OutCount=0;
	int npok = info.np - info.nfluidout;
	if(info.full){
		if(info.Pos) fread((char*)info.Pos,sizeof(tfloat3)*npok,1,BI2FilePnt); 
			else fseek(BI2FilePnt,sizeof(tfloat3)*npok,SEEK_CUR);
		if(info.Vel) fread((char*)info.Vel,sizeof(tfloat3)*npok,1,BI2FilePnt); 
			else fseek(BI2FilePnt,sizeof(tfloat3)*npok,SEEK_CUR);
		if(info.Rhop) fread((char*)info.Rhop,sizeof(float)*npok,1,BI2FilePnt); 
			else fseek(BI2FilePnt,sizeof(float)*npok,SEEK_CUR);
		if(rendi){
			if(info.Pos)fun::ReverseByteOrder((int*)info.Pos,npok*3);
			if(info.Vel)fun::ReverseByteOrder((int*)info.Vel,npok*3);
			if(info.Rhop)fun::ReverseByteOrder((int*)info.Rhop,npok);
		}
	}
	else{
		unsigned nmov=npok-info.nfixed;
		if(info.Pos)fread((char*)(info.Pos+info.nfixed),sizeof(tfloat3)*nmov,1,BI2FilePnt); 
			else fseek(BI2FilePnt,sizeof(tfloat3)*nmov,SEEK_CUR);
		if(info.Vel)fread((char*)(info.Vel+info.nfixed),sizeof(tfloat3)*nmov,1,BI2FilePnt); 
			else fseek(BI2FilePnt,sizeof(tfloat3)*nmov,SEEK_CUR);
		if(info.Rhop)fread((char*)info.Rhop,sizeof(float)*npok,1,BI2FilePnt);              
			else fseek(BI2FilePnt,sizeof(float)*npok,SEEK_CUR);
		if(rendi){
			if(info.Pos)fun::ReverseByteOrder((int*)(info.Pos+info.nfixed),nmov*3);
			if(info.Vel)fun::ReverseByteOrder((int*)(info.Vel+info.nfixed),nmov*3);
			if(info.Rhop)fun::ReverseByteOrder((int*)info.Rhop,npok);
		}
	}
	fclose(BI2FilePnt);
	return ;
}
void BI2Reader::PrintInfo()
{
	printf("Format\t\t:\t%d\n",info.fmt);
	if (info.bitorder == 1)
		printf("Bit Order\t:\t%s\n","BigEndian");
	else printf("Bit Order\t:\t%s\n","LittleEndian");
	if (info.full)
		printf("is Full\t\t:\tTrue\n");
	else printf("is Full\t\t:\tFalse\n");
	if (info.data2d)
		printf("Dimension\t:\t2D\n");
	else printf("Dimension\t:\t3D\n");
	if(info.full){
		printf("Dis bet Part\t:\t%f\n",info.dp);
		printf("h\t\t:\t%f\n",info.h);
		printf("b\t\t:\t%f\n",info.b);
		printf("rhop0\t\t:\t%f\n",info.rhop0);
		printf("gamma\t\t:\t%f\n",info.gamma);
		printf("massbound\t:\t%f\n",info.massbound);
		printf("massfluid\t:\t%f\n",info.massfluid);
	}
	printf("Particle Number\t:\t%d\n",info.np);
	printf("Fixed Number\t:\t%d\n",info.nfixed);
	printf("Floating Number\t:\t%d\n",info.nfloat);
	printf("Moving Number\t:\t%d\n",info.nmoving);
	printf("FluidOut Number\t:\t%d\n",info.nfluidout);
	printf("Time Step\t:\t%f\n",info.time);
}