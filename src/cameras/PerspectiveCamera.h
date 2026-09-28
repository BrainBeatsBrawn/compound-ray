#pragma once

#include "DataRecordCamera.h"
#include "PerspectiveCameraDataTypes.h"

namespace cray
{
    struct PerspectiveCamera : public cray::DataRecordCamera<cray::PerspectiveCameraData>
    {
        PerspectiveCamera (const std::string name);
        ~PerspectiveCamera();

        const char* getEntryFunctionName() const { return "__raygen__pinhole"; }

        // Sets the field of view (FOV) by taking the vertical FOV, in degrees.
        void setYFOV (float yFov);
        // Sets the field of view (FOV) by taking the horizontal FOV, in degrees.
        void setXFOV (float xFov);
        // Sets the aspect ratio of the camera
        void setAspectRatio (float r);

    private:
        float aspectRatio = 1.0f;// Width to height
        float fromDegrees (float d);
    };

} // namespace
