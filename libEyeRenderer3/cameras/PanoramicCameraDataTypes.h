#pragma once

#include "GenericCameraDataTypes.h"

namespace cray
{
    struct PanoramicCameraData
    {
        float startRadius = 0.0f;
        inline bool operator== (const PanoramicCameraData& other) { return (this->startRadius == other.startRadius); }
    };
}

// A typedef for a RaygenPosedContainer containing a PanoramicCameraData
typedef cray::RaygenPosedContainer<cray::PanoramicCameraData> PanoramicCameraPosedData;
