#pragma once

#include "DataRecordCamera.h"
#include "PanoramicCameraDataTypes.h"

namespace cray
{
    struct PanoramicCamera : public cray::DataRecordCamera<cray::PanoramicCameraData>
    {
        PanoramicCamera (const std::string name);
        ~PanoramicCamera();
        void setStartRadius (float d);
        const char* getEntryFunctionName() const { return "__raygen__panoramic"; }
    };

} // namespace
