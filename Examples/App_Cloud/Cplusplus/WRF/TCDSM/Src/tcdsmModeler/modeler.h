#ifndef MODELER_H
#define MODELER_H

#include "export.h"

namespace TCDSM {
    namespace Modeler {

        class TCDSM_MODELER_EXPORT AbstractModeler
        {
        public:
            AbstractModeler();
            virtual ~AbstractModeler() {}
            virtual bool execute() = 0;

        protected:
        };

    }
}

#endif // MODELER_H
