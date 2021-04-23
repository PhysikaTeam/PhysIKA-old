//
// Created by breezelee on 16-10-22.
//

#ifndef TCDSM_MODEL_PARTICLES_H
#define TCDSM_MODEL_PARTICLES_H


#include <osg/Object>
#include <osg/Array>
#include <tcdsmModel/export.h>
#include <tcdsmModel/config.h>


namespace TCDSM {

    namespace Render {
        class MFSCRGeometry;
    }
    namespace Model
    {
        struct TCDSM_MODEL_EXPORT Particle {
            unsigned int index;
            real mass;   //m  -> mass;
            real invMass;   //im -> invMass;
            real radius;   //r  -> radius
            real distance2Camera;   //dc -> distance to camera
            real extinction;   //ex -> extinction
            vec3 position0;   //p0 -> position 0
            vec3 position;   //p  -> position
            vec3 oldPosition;   //op -> oldPosition
            vec3 lastPosition;   //lp -> last position
            vec3 velocity;   //v  -> velocity
            vec3 acceleration;   //a  -> acceleration
            vec4 color;   //c  -> color;
            vec4 incident;   //ic -> incident
            vec4 emergent;   //em -> emergent
        };

        class TCDSM_MODEL_EXPORT Particles :public osg::Object
        {
        public:
            Particles(const unsigned int& size = 0);

            Particles(const Particles& particle, const osg::CopyOp& copyop = osg::CopyOp::DEEP_COPY_ALL);

            virtual osg::Object* cloneType() const { return new Particles(); }
            virtual osg::Object* clone(const osg::CopyOp& copyop) const { return new Particles(*this, copyop); }
            virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const Particles*>(obj) != NULL; }
            virtual const char* libraryName() const { return "TCDSM::Model"; }
            virtual const char* className() const { return "Particles"; }

            //    //设置线程安全
            //    virtual void setThreadSafeRefUnref(bool threadSafe);
            //    virtual void compileGLObjects(RenderInfo& renderInfo)const
            //增加一个结点

            void addVertex(const vec3& vertex);
            void addVertex(const Particles& data, const unsigned int& index);
            void addVertex(const Particle& particle);
            //设置结点
            inline void setParticleData(const unsigned int& i, const vec3& vertex);

            //change a particle
            inline void setParticle(const unsigned int& index, const Particle& particle);
            inline Particle getParticle(const unsigned int& index) const;

            inline       real& getMass(const unsigned int& i);
            inline const real& getMass(const unsigned int& i)const;
            inline       void  setMass(const unsigned int& i, const real mass);
            inline const real& getInvMass(const unsigned int& i)const;

            inline       real& getRadius(const unsigned int& i);
            inline const real& getRadius(const unsigned int& i) const;
            void  setRadius(const unsigned int& i, const real radius);

            inline       real& getDistanceToCamera(const unsigned int& i);
            inline const real& getDistanceToCamera(const unsigned int& i) const;
            inline       void  setDistanceToCamera(const unsigned int& i, const real& distance);

            inline       real& getExtinction(const unsigned int& i);
            inline const real& getExtinction(const unsigned int& i) const;
            inline       void  setExtinction(const unsigned int& i, const real& extinction);
            inline       real* setExtinction();

            inline       vec3& getPosition(const unsigned int& i);
            inline const vec3& getPosition(const unsigned int& i) const;
            inline       void  setPosition(const unsigned int& i, const vec3& position);
            inline  vec3Array* getPositionArray();

            inline       vec3& getPosition0(const unsigned int& i);
            inline const vec3& getPosition0(const unsigned int& i) const;
            inline       void  setPosition0(const unsigned int& i, const vec3& position0);

            inline       vec3& getLastPosition(const unsigned int& i);
            inline const vec3& getLastPosition(const unsigned int& i) const;
            inline       void  setLastPosition(const unsigned int& i, const vec3& lastPosition);

            inline       vec3& getOldPosition(const unsigned int& i);
            inline const vec3& getOldPosition(const unsigned int& i) const;
            inline       void  setOldPosition(const unsigned int& i, const vec3& oldPosition);

            inline       vec3& getVelocity(const unsigned int& i);
            inline const vec3& getVelocity(const unsigned int& i)const;
            inline       void  setVelocity(const unsigned int& i, const vec3& velocity);

            inline       vec3& getAcceleration(const unsigned int& i);
            inline const vec3& getAcceleration(const unsigned int& i)const;
            inline       void  setAcceleration(const unsigned int& i, const vec3& acceleration);

            inline       vec4& getColor(const unsigned int& i);
            inline const vec4& getColor(const unsigned int& i) const;
            inline       void  setColor(const unsigned int& i, const vec4& color);
            inline  vec4Array* getColorArray();

            inline       vec4& getIncident(const unsigned int& i);
            inline const vec4& getIncident(const unsigned int& i) const;
            inline       void  setIncident(const unsigned int& i, const vec4& incident);
            inline  vec4Array* getIncidentArray();

            inline       vec4& getEmergent(const unsigned int& i);
            inline const vec4& getEmergent(const unsigned int& i) const;
            inline       void  setEmergent(const unsigned int& i, const vec4& color);
            inline  vec4Array* getEmergentArray();

            inline void resize(const unsigned int& newsize);
            inline void reserve(const unsigned int& newsize);
            inline void release();

            const boundingBox getBound()const;

            inline unsigned int getNumberOfParticles()const;
            inline unsigned int size() const;

            void setCameraPosition(const vec3& cameraPosition);

            //        bool isLighted(){return _lighted;}
            //        bool setlight(bool flag){ _lighted = flag;}

                    // if a < b return true;
            //    inline bool greatThan(const unsigned int &a, const unsigned int &b) const;
            //    inline bool lessThan (const unsigned int &a, const unsigned int &b) const;
            //    inline void swap(const unsigned int &a, const unsigned int &b);
                    //排序操作
            enum SortOrder {
                ASCENDING,
                DESCENDING
            };
            void sort(const SortOrder order);

            //读写操作
            enum FILETYPE {
                CLOUD_STATIC = 0,
                CLOUD_DYNAMIC = 1,
                CLOUD_STATIC_EXTINCTION = 1 << 1
            };

            virtual bool read(const std::string& path = "", const unsigned int& dataType = CLOUD_STATIC);
            virtual void save(const std::string& path = "", const unsigned int& dataType = CLOUD_STATIC);

        protected:
            virtual ~Particles();

        protected:
            mutable bool _isDirty;
            //bool _lighted;
            mutable boundingBox _boundingBox;

            osg::ref_ptr<uintArray>  _index;

            osg::ref_ptr<realArray>  _masses;
            osg::ref_ptr<realArray>  _invMasses;
            osg::ref_ptr<realArray>  _radius;
            osg::ref_ptr<realArray>  _distance2Cam;
            osg::ref_ptr<realArray>  _extinction;

            osg::ref_ptr<vec3Array>  _position0;
            osg::ref_ptr<vec3Array>  _position;
            osg::ref_ptr<vec3Array>  _oldPosition;
            osg::ref_ptr<vec3Array>  _lastPosition;
            osg::ref_ptr<vec3Array>  _velocity;
            osg::ref_ptr<vec3Array>  _acceleration;
            osg::ref_ptr<vec4Array>  _colors;
            osg::ref_ptr<vec4Array>  _incident;
            osg::ref_ptr<vec4Array>  _emergent;
            friend class Render::MFSCRGeometry;
        };


    }//end namespace Model

} //end of namespace TCDSM


#endif //TCDSM_PARTICLES_H
