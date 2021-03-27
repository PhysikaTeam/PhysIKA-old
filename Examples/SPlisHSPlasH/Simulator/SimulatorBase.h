#ifndef __SimulatorBase_h__
#define __SimulatorBase_h__

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/Utilities/SceneLoader.h"
#include "SPlisHSPlasH/TimeStep.h"
#include "SPlisHSPlasH/FluidModel.h"
#include "ParameterObject.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "SPlisHSPlasH/BoundaryModel_Koschier2017.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"
#include "SPlisHSPlasH/TriangleMesh.h"
#include "BoundarySimulator.h"

namespace SPH
{
	class Simulator_GUI_Base;

	class SimulatorBase : public GenParam::ParameterObject
	{
	public: 
		struct SimulationMethod
		{
			short simulationMethod = 0;
			TimeStep *simulation = NULL;
			FluidModel model;
		};

	protected:
		unsigned int m_numberOfStepsPerRenderUpdate;
		std::string m_exePath;
		std::string m_stateFile;
		std::string m_outputPath;
		std::string m_sceneFile;
		bool m_useParticleCaching;
		bool m_useGUI;
		bool m_isStaticScene;
		Utilities::SceneLoader::Scene m_scene;
		int m_renderWalls;
		bool m_doPause;
		Real m_pauseAt;
		Real m_stopAt;
		bool m_enablePartioExport;
		bool m_enableVTKExport;
		bool m_enableVTIExport;
		bool m_enableRigidBodyVTKExport;
		bool m_enableRigidBodyExport;
		bool m_enableStateExport;
		Real m_framesPerSecond;
		Real m_framesPerSecondState;
		std::string m_particleAttributes;
		std::unique_ptr<Utilities::SceneLoader> m_sceneLoader;
		Real m_nextFrameTime;
		Real m_nextFrameTimeState;
		bool m_firstState;
		unsigned int m_frameCounter;
		bool m_isFirstFrame;
		bool m_isFirstFrameVTK;
		std::vector<std::string> m_colorField;
		std::vector<int> m_colorMapType;
		std::vector<Real> m_renderMaxValue;
		std::vector<Real> m_renderMinValue;
		float const* m_colorMapBuffer;
		unsigned int m_colorMapLength;
		BoundarySimulator *m_boundarySimulator;
		Simulator_GUI_Base *m_gui;
		int m_argc;
		std::vector<char*> m_argv_vec;
		char **m_argv;
		std::string m_windowName;
		std::vector<std::string> m_paramTokens;
		std::function<void()> m_timeStepCB;
		std::vector<float> m_scalarField;
		//-----vti����-----
		std::vector<Vector3r> m_voxelPos;
		std::vector<Real> m_voxelDensity;
		std::vector<Real> m_voxelVelocity;
		unsigned int m_voxelPointsIndex;
		bool m_isInitWriteParticlesVTI;
		//�洢x��y��z��ά��С
		std::vector<int> m_voxelSize;
		//-----------------
#ifdef DL_OUTPUT
		Real m_nextTiming;
#endif

		virtual void initParameters();

		void initFluidData();
		void createFluidBlocks(std::map<std::string, unsigned int> &fluidIDs, std::vector<std::vector<Vector3r>> &fluidParticles, std::vector<std::vector<Vector3r>> &fluidVelocities);
		void createEmitters();
		void createAnimationFields();
		void buildModel();
		void setCommandLineParameter();
		void setCommandLineParameter(GenParam::ParameterObject *paramObj);

	public:
		static int PAUSE;
		static int PAUSE_AT;
		static int STOP_AT;
		static int NUM_STEPS_PER_RENDER;
		static int PARTIO_EXPORT;
		static int VTK_EXPORT;
		static int VTI_EXPORT;
		static int RB_VTK_EXPORT;
		static int RB_EXPORT;
		static int DATA_EXPORT_FPS;
		static int PARTICLE_EXPORT_ATTRIBUTES;
		static int STATE_EXPORT;
		static int STATE_EXPORT_FPS;
		static int RENDER_WALLS;
		
		static int ENUM_WALLS_NONE;
		static int ENUM_WALLS_PARTICLES_ALL;
		static int ENUM_WALLS_PARTICLES_NO_WALLS;
		static int ENUM_WALLS_GEOMETRY_ALL;
		static int ENUM_WALLS_GEOMETRY_NO_WALLS;

		SimulatorBase();
		SimulatorBase(const SimulatorBase&) = delete;
        SimulatorBase& operator=(const SimulatorBase&) = delete;
		virtual ~SimulatorBase();

		void run();
		void init(std::vector<std::string> argv, const std::string &windowName);
		void init(int argc, char **argv, const std::string &windowName);
        void initSimulation();
		void runSimulation();
		void cleanup();

		void reset();
		void timeStep();
		bool timeStepNoGUI();

		void setTimeStepCB(std::function<void()> const& callBackFct) { m_timeStepCB = callBackFct; }

		static void particleInfo(std::vector<std::vector<unsigned int>> &particles);

		std::string real2String(const Real r);
		void initDensityMap(std::vector<Vector3r> &x, std::vector<unsigned int> &faces, const Utilities::SceneLoader::BoundaryData *boundaryData, const bool md5, const bool isDynamic, BoundaryModel_Koschier2017 *boundaryModel);
		void initVolumeMap(std::vector<Vector3r> &x, std::vector<unsigned int> &faces, const Utilities::SceneLoader::BoundaryData *boundaryData, const bool md5, const bool isDynamic, BoundaryModel_Bender2019 *boundaryModel);

		void readParameters();
		void particleExport();
		void rigidBodyExport();
		void writeParticlesPartio(const std::string &fileName, FluidModel *model);
		void writeParticlesVTK(const std::string &fileName, FluidModel *model);

		//--------2021/3/9-----------
		void initWriteParticlesVTI();
		void writeParticlesVTI(const std::string& fileName);
		std::string base64_encode(unsigned char const* bytes_to_encode, size_t in_len);
		FORCE_INLINE Vector3r& getVoxelPos(const unsigned int i)
		{
			return m_voxelPos[i];
		}
		FORCE_INLINE Real& getVoxelDensity(const unsigned int i)
		{
			return m_voxelDensity[i];
		}
		FORCE_INLINE Real& getVoxelVelocity(const unsigned int i)
		{
			return m_voxelVelocity[i];
		}
		unsigned int getFrameCounter() const { return m_frameCounter; }
		//---------------------------
		void writeRigidBodiesBIN(const std::string &exportPath);
		void writeRigidBodiesVTK(const std::string &exportPath);
		void step();

		void saveState(const std::string& stateFile = "");
		void loadStateDialog();
		void loadState(const std::string &stateFile);
		void writeFluidParticlesState(const std::string &fileName, FluidModel *model);
		void readFluidParticlesState(const std::string &fileName, FluidModel *model);
		void writeBoundaryState(const std::string &fileName, BoundaryModel *model);
		void readBoundaryState(const std::string &fileName, BoundaryModel *model);
		void writeParameterState(BinaryFileWriter &binWriter);
		void readParameterState(BinaryFileReader &binReader);
		void writeParameterObjectState(BinaryFileWriter &binWriter, GenParam::ParameterObject *paramObj);
		void readParameterObjectState(BinaryFileReader &binReader, GenParam::ParameterObject *paramObj);

		void updateBoundaryParticles(const bool forceUpdate);
		void updateDMVelocity();
		void updateVMVelocity();

		std::vector<float>& getScalarField() { return m_scalarField; }
		void updateScalarField();
		void determineMinMaxOfScalarField();

		static void loadObj(const std::string &filename, TriangleMesh &mesh, const Vector3r &scale);

		Utilities::SceneLoader *getSceneLoader() { return m_sceneLoader.get(); }

		const std::string& getExePath() const { return m_exePath; }
		void setSceneFile(const std::string& file) { m_sceneFile = file; }
		const std::string& getSceneFile() const { return m_sceneFile; }

		Utilities::SceneLoader::Scene& getScene() { return m_scene; }

		bool getUseParticleCaching() const { return m_useParticleCaching; }
		void setUseParticleCaching(bool val) { m_useParticleCaching = val; }
		bool getUseGUI() const { return m_useGUI; }
		void setUseGUI(bool val) { m_useGUI = val; }

		const std::string& getColorField(const unsigned int fluidModelIndex) {	return m_colorField[fluidModelIndex]; }
		void setColorField(const unsigned int fluidModelIndex, const std::string& fieldName) { m_colorField[fluidModelIndex] = fieldName; }

		int getColorMapType(const unsigned int fluidModelIndex) const { return m_colorMapType[fluidModelIndex]; }
		void setColorMapType(const unsigned int fluidModelIndex, int val) { m_colorMapType[fluidModelIndex] = val; }
		Real getRenderMaxValue(const unsigned int fluidModelIndex) const { return m_renderMaxValue[fluidModelIndex]; }
		void setRenderMaxValue(const unsigned int fluidModelIndex, Real val) { m_renderMaxValue[fluidModelIndex] = val; }
		Real getRenderMinValue(const unsigned int fluidModelIndex) const { return m_renderMinValue[fluidModelIndex]; }
		void setRenderMinValue(const unsigned int fluidModelIndex, Real val) { m_renderMinValue[fluidModelIndex] = val; }
		std::string getOutputPath() const { return m_outputPath; }

		// VTK expects big endian
		template<typename T>
		inline void swapByteOrder(T*v)
		{
			constexpr size_t n = sizeof(T);
			uint8_t * bytes = reinterpret_cast<uint8_t*>(v);
			for (unsigned int c = 0u; c < n / 2; c++)
				std::swap(bytes[c], bytes[n - c - 1]);
		}

		std::string getStateFile() const { return m_stateFile; }
		void setStateFile(std::string val) { m_stateFile = val; }

		SPH::BoundarySimulator * getBoundarySimulator() const { return m_boundarySimulator; }
		void setBoundarySimulator(SPH::BoundarySimulator * val) { m_boundarySimulator = val; }
		SPH::Simulator_GUI_Base * getGui() const { return m_gui; }
		void setGui(SPH::Simulator_GUI_Base * val) { m_gui = val; }
		bool isStaticScene() const { return m_isStaticScene; }
	};
}
 
#endif
