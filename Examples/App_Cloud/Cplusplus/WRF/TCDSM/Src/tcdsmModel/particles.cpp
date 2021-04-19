#include <tcdsmModel/particles.h>
#include <fstream>
#include <queue>

using namespace TCDSM::Model;

Particles::Particles(const unsigned int &size)
        :Object()
        ,_isDirty(true)
//        ,_lighted(false)
        ,_boundingBox()
        ,_index          (new uintArray(size))
        ,_masses         (new realArray(size))
        ,_invMasses      (new realArray(size))
        ,_radius         (new realArray(size))
        ,_distance2Cam   (new realArray(size))
        ,_extinction     (new realArray(size))
        ,_position0      (new vec3Array(size))
        ,_position       (new vec3Array(size))
        ,_oldPosition    (new vec3Array(size))
        ,_lastPosition   (new vec3Array(size))
        ,_velocity       (new vec3Array(size))
        ,_acceleration   (new vec3Array(size))
        ,_colors         (new vec4Array(size))
        ,_incident       (new vec4Array(size))
        ,_emergent       (new vec4Array(size))
{
    setDataVariance(DYNAMIC);
    _index->setDataVariance(DYNAMIC);
    _colors->setDataVariance(DYNAMIC);
    _radius->setDataVariance(DYNAMIC);
    _extinction->setDataVariance(DYNAMIC);
    _distance2Cam -> setDataVariance(DYNAMIC);

	for (unsigned int i = 0; i < size; ++i) {
		(*_index)[i] = i;
	}
}

Particles::Particles(const Particles &particles, const osg::CopyOp &copyop)
        : Object(particles,copyop)
        ,_isDirty(particles._isDirty)
//        ,_lighted(particles._lighted)
        ,_boundingBox(particles._boundingBox)
        ,_index          ((uintArray*)copyop(particles._index       .get() ))
        ,_masses         ((realArray*)copyop(particles._masses      .get() ))
        ,_invMasses      ((realArray*)copyop(particles._invMasses   .get() ))
        ,_radius         ((realArray*)copyop(particles._radius      .get() ))
        ,_distance2Cam   ((realArray*)copyop(particles._distance2Cam.get() ))
        ,_extinction     ((realArray*)copyop(particles._extinction  .get() ))
        ,_position0      ((vec3Array*)copyop(particles._position0   .get() ))
        ,_position       ((vec3Array*)copyop(particles._position    .get() ))
        ,_oldPosition    ((vec3Array*)copyop(particles._oldPosition .get() ))
        ,_lastPosition   ((vec3Array*)copyop(particles._lastPosition.get() ))
        ,_velocity       ((vec3Array*)copyop(particles._velocity    .get() ))
        ,_acceleration   ((vec3Array*)copyop(particles._acceleration.get() ))
        ,_colors         ((vec4Array*)copyop(particles._colors      .get() ))
        ,_incident       ((vec4Array*)copyop(particles._incident    .get() ))
        ,_emergent       ((vec4Array*)copyop(particles._emergent    .get() ))
{
}

//void LY_Model::Particles::setThreadSafeRefUnref(bool threadSafe) {
//    osg::Object::setThreadSafeRefUnref(threadSafe);
//    _index        ->setThreadSafeRefUnref(threadSafe);
//    _masses       ->setThreadSafeRefUnref(threadSafe);
//    _invMasses    ->setThreadSafeRefUnref(threadSafe);
//    _radius       ->setThreadSafeRefUnref(threadSafe);
//    _distance2Cam ->setThreadSafeRefUnref(threadSafe);
//    _extinction   ->setThreadSafeRefUnref(threadSafe);
//    _position0    ->setThreadSafeRefUnref(threadSafe);
//    _position     ->setThreadSafeRefUnref(threadSafe);
//    _oldPosition  ->setThreadSafeRefUnref(threadSafe);
//    _lastPosition ->setThreadSafeRefUnref(threadSafe);
//    _velocity     ->setThreadSafeRefUnref(threadSafe);
//    _acceleration ->setThreadSafeRefUnref(threadSafe);
//    _colors       ->setThreadSafeRefUnref(threadSafe);
//    _incident     ->setThreadSafeRefUnref(threadSafe);
//    _emergent     ->setThreadSafeRefUnref(threadSafe);
//}

inline Particles::~Particles()
{
    release();
}

void Particles::addVertex(const vec3& vertex)
{
    _isDirty = true;
    _index->push_back((unsigned int)_index->size());
    _masses->push_back(1.0);
    _invMasses->push_back(1.0);
    _radius->push_back(1.0);
    _distance2Cam->push_back(0.0);
    _extinction->push_back(0.0);
    _position0->push_back(vertex);
    _position->push_back(vertex);
    _oldPosition->push_back(vertex);
    _lastPosition->push_back(vertex);
    _velocity->push_back(vec3(0.0, 0.0, 0.0));
    _acceleration->push_back(vec3(0.0, 0.0, 0.0));
    _colors->push_back(vec4(1.0, 1.0, 1.0, 1.0));
    _incident->push_back(vec4(1.0, 1.0, 1.0, 1.0));
    _emergent->push_back(vec4(1.0, 1.0, 1.0, 1.0));
}

void Particles::addVertex(const Particles& data, const unsigned int& index)
{
    _isDirty = true;
    _index->push_back((unsigned int)_index->size());
    _masses->push_back(data.getMass(index));
    _invMasses->push_back(data.getInvMass(index));
    _radius->push_back(data.getRadius(index));
    _distance2Cam->push_back(data.getDistanceToCamera(index));
    _extinction->push_back(data.getExtinction(index));
    _position0->push_back(data.getPosition0(index));
    _position->push_back(data.getPosition(index));
    _oldPosition->push_back(data.getOldPosition(index));
    _lastPosition->push_back(data.getLastPosition(index));
    _velocity->push_back(data.getVelocity(index));
    _acceleration->push_back(data.getAcceleration(index));
    _colors->push_back(data.getColor(index));
    _incident->push_back(data.getIncident(index));
    _emergent->push_back(data.getEmergent(index));
}

void Particles::addVertex(const Particle& p)
{
    _isDirty = true;
    _index->push_back((unsigned int)_index->size());
    _masses->push_back(p.mass);
    _invMasses->push_back(p.invMass);
    _radius->push_back(p.radius);
    _distance2Cam->push_back(p.distance2Camera);
    _extinction->push_back(p.extinction);
    _position0->push_back(p.position0);
    _position->push_back(p.position);
    _oldPosition->push_back(p.oldPosition);
    _lastPosition->push_back(p.lastPosition);
    _velocity->push_back(p.velocity);
    _acceleration->push_back(p.acceleration);
    _colors->push_back(p.color);
    _incident->push_back(p.incident);
    _emergent->push_back(p.emergent);
}

inline void Particles::setParticleData(const unsigned int& i, const vec3& vertex)
{
    if (i > _index->size())
        return;

    //        (*_masses      )[(*_index)[i]] = (1.0);
    //        (*_invMasses   )[(*_index)[i]] = (1.0);
    //        (*_radius      )[(*_index)[i]] = (1.0);
    //        (*_distance2Cam)[(*_index)[i]] = (0.0);
    //        (*_extinction  )[(*_index)[i]] = (0.0);
    //        (*_position0   )[(*_index)[i]] = (vertex);
    (*_position)[(*_index)[i]] = (vertex);
    //        (*_oldPosition )[(*_index)[i]] = (vertex);
    //        (*_lastPosition)[(*_index)[i]] = (vertex);
    //        (*_velocity    )[(*_index)[i]] = (vec3(0.0,0.0,0.0));
    //        (*_acceleration)[(*_index)[i]] = (vec3(0.0,0.0,0.0));
    //        (*_colors      )[(*_index)[i]] = (vec4(1.0,1.0,1.0,1.0));
    //        (*_incident    )[(*_index)[i]] = (vec4(1.0,1.0,1.0,1.0));
    //        (*_emergent    )[(*_index)[i]] = (vec4(1.0,1.0,1.0,1.0));
}

void Particles::setParticle(const unsigned int& index, const Particle& p)
{
    _isDirty = true;
    if (_index->size() > index)
        return;
    (*_masses)[(*_index)[index]] = p.mass;
    (*_invMasses)[(*_index)[index]] = p.invMass;
    (*_radius)[(*_index)[index]] = p.radius;
    (*_distance2Cam)[(*_index)[index]] = p.distance2Camera;
    (*_extinction)[(*_index)[index]] = p.extinction;
    (*_position0)[(*_index)[index]] = p.position0;
    (*_position)[(*_index)[index]] = p.position;
    (*_oldPosition)[(*_index)[index]] = p.oldPosition;
    (*_lastPosition)[(*_index)[index]] = p.lastPosition;
    (*_velocity)[(*_index)[index]] = p.velocity;
    (*_acceleration)[(*_index)[index]] = p.acceleration;
    (*_colors)[(*_index)[index]] = p.color;
    (*_incident)[(*_index)[index]] = p.incident;
    (*_emergent)[(*_index)[index]] = p.emergent;

}

Particle Particles::getParticle(const unsigned int& index) const
{
    Particle p;
    p.index = index;
    p.mass = (*_masses)[(*_index)[index]];
    p.invMass = (*_invMasses)[(*_index)[index]];
    p.radius = (*_radius)[(*_index)[index]];
    p.distance2Camera = (*_distance2Cam)[(*_index)[index]];
    p.extinction = (*_extinction)[(*_index)[index]];
    p.position0 = (*_position0)[(*_index)[index]];
    p.position = (*_position)[(*_index)[index]];
    p.oldPosition = (*_oldPosition)[(*_index)[index]];
    p.lastPosition = (*_lastPosition)[(*_index)[index]];
    p.velocity = (*_velocity)[(*_index)[index]];
    p.acceleration = (*_acceleration)[(*_index)[index]];
    p.color = (*_colors)[(*_index)[index]];
    p.incident = (*_incident)[(*_index)[index]];
    p.emergent = (*_emergent)[(*_index)[index]];
    return p;
}

inline void Particles::resize(const unsigned int& newsize)
{
    unsigned  int oldsize = (unsigned int)(_index->size());
    _index->resize(newsize);
    _masses->resize(newsize);
    _invMasses->resize(newsize);
    _radius->resize(newsize);
    _distance2Cam->resize(newsize);
    _extinction->resize(newsize);
    _position0->resize(newsize);
    _position->resize(newsize);
    _oldPosition->resize(newsize);
    _lastPosition->resize(newsize);
    _velocity->resize(newsize);
    _acceleration->resize(newsize);
    _colors->resize(newsize);
    _incident->resize(newsize);
    _emergent->resize(newsize);
    for (unsigned int i = oldsize; i < newsize; ++i) {
        (*_index)[i] = i;
    }
}

inline void Particles::reserve(const unsigned int& newsize)
{
    unsigned  int oldsize = (unsigned int)(_index->size());
    _index->resize(newsize);
    _masses->resize(newsize);
    _invMasses->resize(newsize);
    _radius->resize(newsize);
    _distance2Cam->resize(newsize);
    _extinction->resize(newsize);
    _position0->resize(newsize);
    _position->resize(newsize);
    _oldPosition->resize(newsize);
    _lastPosition->resize(newsize);
    _velocity->resize(newsize);
    _acceleration->resize(newsize);
    _colors->resize(newsize);
    _incident->resize(newsize);
    _emergent->resize(newsize);
    for (unsigned int i = oldsize; i < newsize; ++i) {
        (*_index)[i] = i;
    }
}

inline void Particles::release()
{
    _boundingBox.init();
    _index->clear();
    _masses->clear();
    _invMasses->clear();
    _radius->clear();
    _distance2Cam->clear();
    _extinction->clear();
    _position0->clear();
    _position->clear();
    _oldPosition->clear();
    _lastPosition->clear();
    _velocity->clear();
    _acceleration->clear();
    _colors->clear();
    _incident->clear();
    _emergent->clear();
}

//    inline void Particles::swap(const unsigned int &a, const unsigned int &b)
//    {
    //real tmpReal;vec3 tmpVec3; vec4 tmpVec4;
    //tmpReal = (*_masses      )[a]; (*_masses      )[a] = (*_masses      )[b];(*_masses      )[b] = tmpReal;
    //tmpReal = (*_invMasses   )[a]; (*_invMasses   )[a] = (*_invMasses   )[b];(*_invMasses   )[b] = tmpReal;
    //tmpReal = (*_radius      )[a]; (*_radius      )[a] = (*_radius      )[b];(*_radius      )[b] = tmpReal;
    //tmpReal = (*_distance2Cam)[a]; (*_distance2Cam)[a] = (*_distance2Cam)[b];(*_distance2Cam)[b] = tmpReal;
    //tmpReal = (*_extinction  )[a]; (*_extinction  )[a] = (*_extinction  )[b];(*_extinction  )[b] = tmpReal;
    //tmpVec3 = (*_position0   )[a]; (*_position0   )[a] = (*_position0   )[b];(*_position0   )[b] = tmpVec3;
    //tmpVec3 = (*_position    )[a]; (*_position    )[a] = (*_position    )[b];(*_position    )[b] = tmpVec3;
    //tmpVec3 = (*_oldPosition )[a]; (*_oldPosition )[a] = (*_oldPosition )[b];(*_oldPosition )[b] = tmpVec3;
    //tmpVec3 = (*_lastPosition)[a]; (*_lastPosition)[a] = (*_lastPosition)[b];(*_lastPosition)[b] = tmpVec3;
    //tmpVec3 = (*_velocity    )[a]; (*_velocity    )[a] = (*_velocity    )[b];(*_velocity    )[b] = tmpVec3;
    //tmpVec3 = (*_acceleration)[a]; (*_acceleration)[a] = (*_acceleration)[b];(*_acceleration)[b] = tmpVec3;
    //tmpVec4 = (*_colors      )[a]; (*_colors      )[a] = (*_colors      )[b];(*_colors      )[b] = tmpVec4;
    //tmpVec4 = (*_incident    )[a]; (*_incident    )[a] = (*_incident    )[b];(*_incident    )[b] = tmpVec4;
    //tmpVec4 = (*_emergent    )[a]; (*_emergent    )[a] = (*_emergent    )[b];(*_emergent    )[b] = tmpVec4;
//        unsigned  int i = (*_index)[a];
//        (*_index)[a] = (*_index)[b];
//        (*_index)[b] = i;
//
//    }

inline vec4& Particles::getColor(const unsigned int& i)
{
    return (*_colors)[(*_index)[i]];
}

inline const vec4& Particles::getColor(const unsigned int& i) const
{
    return (*_colors)[(*_index)[i]];
}

inline void Particles::setColor(const unsigned int& i, const vec4& color)
{
    (*_colors)[(*_index)[i]] = color;
}

vec4Array* Particles::getColorArray()
{
    return _colors.get();
}

inline vec4& Particles::getIncident(const unsigned int& i)
{
    return (*_incident)[(*_index)[i]];
}

inline const vec4& Particles::getIncident(const unsigned int& i) const
{
    return (*_incident)[(*_index)[i]];
}

inline void Particles::setIncident(const unsigned int& i, const vec4& incident)
{
    (*_incident)[(*_index)[i]] = incident;
}

vec4Array* Particles::getIncidentArray()
{
    return _incident.get();
}

inline vec4& Particles::getEmergent(const unsigned int& i)
{
    return (*_emergent)[(*_index)[i]];
}

inline const vec4& Particles::getEmergent(const unsigned int& i) const
{
    return (*_emergent)[(*_index)[i]];
}

inline void Particles::setEmergent(const unsigned int& i, const vec4& emergent)
{
    (*_emergent)[(*_index)[i]] = emergent;
}

vec4Array* Particles::getEmergentArray()
{
    return _emergent.get();
}

inline vec3& Particles::getPosition(const unsigned int& i)
{
    _isDirty = true;
    return (*_position)[(*_index)[i]];
}

inline const vec3& Particles::getPosition(const unsigned int& i) const
{
    return (*_position)[(*_index)[i]];
}

inline void Particles::setPosition(const unsigned int& i, const vec3& position)
{
    if ((*_position)[(*_index)[i]] != position) 
	{
        (*_position)[(*_index)[i]] = position;

		//printf("(*_index)[i] = %d\n", (*_index)[i]);

        _isDirty = true;
    }
}

inline vec3Array* Particles::getPositionArray()
{
    return _position.get();
}

inline real& Particles::getDistanceToCamera(const unsigned int& i)
{
    return (*_distance2Cam)[(*_index)[i]];
}

inline const real& Particles::getDistanceToCamera(const unsigned int& i) const
{
    return (*_distance2Cam)[(*_index)[i]];
}

inline void Particles::setDistanceToCamera(const unsigned int& i, const real& distance)
{
    (*_distance2Cam)[(*_index)[i]] = distance;
}


inline vec3& Particles::getPosition0(const unsigned int& i)
{
    return (*_position0)[(*_index)[i]];
}

inline const vec3& Particles::getPosition0(const unsigned int& i) const
{
    return (*_position0)[(*_index)[i]];
}

inline void Particles::setPosition0(const unsigned int& i, const vec3& position0)
{
    (*_position0)[(*_index)[i]] = position0;
}

inline vec3& Particles::getLastPosition(const unsigned int& i)
{
    return (*_lastPosition)[(*_index)[i]];
}

inline const vec3& Particles::getLastPosition(const unsigned int& i)const
{
    return (*_lastPosition)[(*_index)[i]];
}

inline void Particles::setLastPosition(const unsigned int& i, const vec3& lastPosition)
{
    (*_lastPosition)[(*_index)[i]] = lastPosition;
}

inline vec3& Particles::getOldPosition(const unsigned int& i)
{
    return (*_oldPosition)[(*_index)[i]];
}

inline const vec3& Particles::getOldPosition(const unsigned int& i) const
{
    return (*_oldPosition)[(*_index)[i]];
}

inline void Particles::setOldPosition(const unsigned int& i, const vec3& oldPosition)
{
    (*_oldPosition)[(*_index)[i]] = oldPosition;
}

inline vec3& Particles::getVelocity(const unsigned int& i)
{
    return (*_velocity)[(*_index)[i]];
}

inline const vec3& Particles::getVelocity(const unsigned int& i)const
{
    return (*_velocity)[(*_index)[i]];
}

inline void Particles::setVelocity(const unsigned int& i, const vec3& velocity)
{
    (*_velocity)[(*_index)[i]] = velocity;
}

inline vec3& Particles::getAcceleration(const unsigned int& i)
{
    return (*_acceleration)[(*_index)[i]];
}

inline const vec3& Particles::getAcceleration(const unsigned int& i)const
{
    return (*_acceleration)[(*_index)[i]];
}

inline void Particles::setAcceleration(const unsigned int& i, const vec3& acceleration)
{
    (*_acceleration)[(*_index)[i]] = acceleration;
}

inline real& Particles::getMass(const unsigned int& i)
{
    return (*_masses)[(*_index)[i]];
}

inline const real& Particles::getMass(const unsigned int& i)const
{
    return (*_masses)[(*_index)[i]];
}

void Particles::setMass(const unsigned int& i, const real mass)
{
    (*_masses)[(*_index)[i]] = mass;

    if (fabs(mass - 0.0) < 0.00000001)
        (*_invMasses)[(*_index)[i]] = (real)1.0 / mass;
    else
        (*_invMasses)[(*_index)[i]] = 0.0;
}

inline const real& Particles::getInvMass(const unsigned int& i)const
{
    return (*_invMasses)[(*_index)[i]];
}

inline real& Particles::getRadius(const unsigned int& i)
{
    return (*_radius)[(*_index)[i]];
}

inline const real& Particles::getRadius(const unsigned int& i) const
{
    return (*_radius)[(*_index)[i]];
}

inline void Particles::setRadius(const unsigned int& i, const real radius)
{
    (*_radius)[(*_index)[i]] = radius;
}

inline real& Particles::getExtinction(const unsigned int& i)
{
    return (*_extinction)[(*_index)[i]];
}

inline const real& Particles::getExtinction(const unsigned int& i) const
{
    return (*_extinction)[(*_index)[i]];
}

inline void Particles::setExtinction(const unsigned int& i, const real& extinction)
{
    (*_extinction)[(*_index)[i]] = extinction;
}

inline real* Particles::setExtinction()
{
    return &((*_extinction)[0]);
}

inline unsigned int Particles::getNumberOfParticles()const
{
    return (unsigned int)_index->size();
}

inline unsigned int Particles::size() const
{
    return (unsigned int)_index->size();
}

bool Particles::read(const std::string &path,const unsigned int &dataType) {
    //todo double type deal
    if(path == "")
        return false;

    std::ifstream infile(path.c_str());
    if(!infile.is_open()){
        return false;
    }

    infile.seekg(0,std::ios::end);
    unsigned long fileSize = infile.tellg();
    infile.seekg(0,std::ios::beg);

    unsigned int psize;
    infile.read((char *)(&psize),sizeof(psize));

    unsigned int dataType2;
    fileSize -= 4;

    unsigned int sizemod = fileSize % (psize * sizeof(float));
    unsigned int sizep = fileSize / (psize * sizeof(float));
    if(0 != sizemod)
    {
        infile.close();
        return false;
    }

    if(sizep == 8 )
    {
        dataType2 = CLOUD_STATIC;
    }
    else if(sizep == 9) {
            if(dataType == CLOUD_STATIC)
                dataType2 = CLOUD_STATIC;
            else
                dataType2 = CLOUD_STATIC_EXTINCTION;
    }
    else if(sizep == 23  ){
        dataType2 = dataType;
    }
    else
    {
        infile.close();
        return false;
    }

    resize(psize);
    infile.read((char *)_position->getDataPointer(),sizeof(float)* psize * 3);
    infile.read((char *)_radius  ->getDataPointer(),sizeof(float)* psize * 1);
    infile.read((char *)_colors  ->getDataPointer(),sizeof(float)* psize * 4);

    for (unsigned int i = 0; i < psize; ++i) {
        (*_position0)[i] = vec3((*_position)[i].x(),(*_position)[i].z(),(*_position)[i].y());
        (*_position)[i] = (*_position0)[i];
    }

    if(dataType2 & CLOUD_STATIC_EXTINCTION)
        infile.read((char *)_extinction->getDataPointer(),sizeof(float)*psize);

    if(dataType2 & CLOUD_DYNAMIC)
    {
        infile.read((char *)_extinction  ->getDataPointer(),sizeof(float)*psize*1);
        infile.read((char *)_masses      ->getDataPointer(),sizeof(float)*psize*1);
        infile.read((char *)_velocity    ->getDataPointer(),sizeof(float)*psize*3);
        infile.read((char *)_oldPosition ->getDataPointer(),sizeof(float)*psize*3);
        infile.read((char *)_lastPosition->getDataPointer(),sizeof(float)*psize*3);
        infile.read((char *)_acceleration->getDataPointer(),sizeof(float)*psize*3);

        float *invmassData = (float *)_invMasses->getDataPointer();
        float *massData    = (float *)_masses   ->getDataPointer();
        osg::Vec3 *p0Data  = (osg::Vec3 *)_position0->getDataPointer();
        osg::Vec3 *pData   = (osg::Vec3 *)_position ->getDataPointer();
        unsigned int *in   = (unsigned int *)_index ->getDataPointer();
        for(unsigned int i = 0; i < psize; ++i){
            in[i] = i;
            invmassData[i] = 1.0/massData[i];
            p0Data[i] = pData[i];
        }
    }
    //TODO read other file
    infile.close();
    _isDirty = true;
    return true;

}

void Particles::save(const std::string &path, const unsigned int &dataType) {
	//TODO double type deal

	if (path == "")
		return;
	std::ofstream outfile(path.c_str(), std::ios_base::binary | std::ios_base::out);
	if (!outfile.is_open())
		return;

	unsigned int pSize = (unsigned int)(_index->size());
	printf("pSize = %u\n", pSize);

	outfile.write((char *)&pSize, sizeof(pSize));
	osg::ref_ptr<vec3Array> pos = new vec3Array;
	pos->resize(pSize);
	for (unsigned int i = 0; i < pSize; ++i)
	{
		(*pos)[i] = vec3((*_position)[i].x(), (*_position)[i].z(), (*_position)[i].y());
		//printf("pos[%d] = %f  %f  %f\n", i, (*pos)[i].x(), (*pos)[i].y(), (*pos)[i].z());
	}
	outfile.write((char *)(pos->getDataPointer()), sizeof(float) * 3 * pSize);
	outfile.write((char *)(_radius->getDataPointer()), sizeof(float) * 1 * pSize);
	outfile.write((char *)(_colors->getDataPointer()), sizeof(float) * 4 * pSize);

	if (dataType & CLOUD_STATIC_EXTINCTION)
		outfile.write((char *)(_extinction->getDataPointer()), sizeof(float) * 1 * pSize);
	//TODO save other files;
	if (dataType & CLOUD_DYNAMIC)
	{
		outfile.write((char *)_extinction->getDataPointer(), sizeof(float) * 1 * pSize);
		outfile.write((char *)_masses->getDataPointer(), sizeof(float) * 1 * pSize);
		outfile.write((char *)_velocity->getDataPointer(), sizeof(float) * 3 * pSize);
		outfile.write((char *)_oldPosition->getDataPointer(), sizeof(float) * 3 * pSize);
		outfile.write((char *)_lastPosition->getDataPointer(), sizeof(float) * 3 * pSize);
		outfile.write((char *)_acceleration->getDataPointer(), sizeof(float) * 3 * pSize);
	}
	outfile.close();
	return;
}

const boundingBox Particles::getBound() const{
    if(_isDirty)
    {
        _boundingBox.init();
        for(unsigned int particleIndex = 0;
            particleIndex < size(); ++particleIndex)
        {
            _boundingBox.expandBy((*_position)[particleIndex]);
        }
        _isDirty = false;
    }
    return _boundingBox;
}

void Particles::setCameraPosition(const vec3 &cameraPosition) {
    const unsigned int size = (const unsigned int)(_position->size());
    for(unsigned int i = 0; i < size; ++i)
    {
        (*_distance2Cam)[i] = (cameraPosition - (*_position)[i]).length2();
    }
}

struct __SortStruct{
    __SortStruct(const real &d,const unsigned int &i):distance(d),index(i){}
    real distance;
    unsigned int index;
    virtual bool operator < (const __SortStruct&b)const{return distance < b.distance ; }
};

struct __SortDsc:public __SortStruct{
    __SortDsc(const real &d, const unsigned int &i):__SortStruct(d,i){}
    virtual bool operator < (const __SortStruct&b)const{
        return distance < b.distance;
    }
};

struct __SortAsc:public __SortStruct{
    __SortAsc(const real &d, const unsigned int &i):__SortStruct(d,i){}
    virtual bool operator < (const __SortStruct&b)const{
        return distance > b.distance;
    }
};

void Particles::sort(const Particles::SortOrder order) {

    //todo sort
    unsigned long pSize = _index->size();
    if(order == ASCENDING)
    {
        std::priority_queue<__SortAsc> sortQueue;
        for (unsigned long i = 0; i < pSize; ++i)
            sortQueue.push(__SortAsc((*_distance2Cam)[i],(unsigned int)i));
        for (unsigned long i = 0; i < pSize; ++i)
        {
            (*_index)[i] = sortQueue.top().index;
            sortQueue.pop();
        }
    }
    if(order == DESCENDING)
    {
        std::priority_queue<__SortDsc> sortQueue;
        for (unsigned long i = 0; i < pSize; ++i)
            sortQueue.push(__SortDsc((*_distance2Cam)[i],(unsigned int)i));
        for (unsigned long i = 0; i < pSize; ++i)
        {
            (*_index)[i] = sortQueue.top().index;
            sortQueue.pop();
        }
    }

}
