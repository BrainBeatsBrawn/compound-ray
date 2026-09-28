#pragma once

#include <string>
#include "DataRecordCamera.h"
#include "OrthographicCameraDataTypes.h"

namespace cray
{
    struct OrthographicCamera : public cray::DataRecordCamera<OrthographicCameraData>
    {
        OrthographicCamera (const std::string name);
        ~OrthographicCamera();

        const char* getEntryFunctionName() const { return "__raygen__orthographic"; }

        void setXYscale (float x, float y);
        void setXYscale (float2 scale) { this->setXYscale (scale.x, scale.y); }
    };

} // namespace
