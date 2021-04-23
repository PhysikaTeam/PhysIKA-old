//=============================================================================
//
//  Implements an IOManager writer module for VTK files
//
//=============================================================================

#ifndef __VTKWRITER_HH__
#define __VTKWRITER_HH__

//=== INCLUDES ================================================================

#include <string>
#include <iosfwd>

#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Core/Utils/SingletonT.hh>
#include <OpenMesh/Core/IO/exporter/BaseExporter.hh>
#include <OpenMesh/Core/IO/writer/BaseWriter.hh>

//== NAMESPACES ===============================================================

namespace OpenMesh {
namespace IO {

//=== IMPLEMENTATION ==========================================================

class OPENMESHDLLEXPORT _VTKWriter_ : public BaseWriter
{
public:
    _VTKWriter_();

    std::string get_description() const override { return "VTK"; }
    std::string get_extensions()  const override { return "vtk"; }

    bool write(const std::string&, BaseExporter&, Options, std::streamsize _precision = 6) const override;
    bool write(std::ostream&, BaseExporter&, Options, std::streamsize _precision = 6) const override;

    size_t binary_size(BaseExporter&, Options) const override { return 0; }
};

//== TYPE DEFINITION ==========================================================

/// Declare the single entity of the OBJ writer
extern _VTKWriter_  __VTKWriterinstance;
OPENMESHDLLEXPORT _VTKWriter_& VTKWriter();

//=============================================================================
} // namespace IO
} // namespace OpenMesh
//=============================================================================
#endif
//=============================================================================
