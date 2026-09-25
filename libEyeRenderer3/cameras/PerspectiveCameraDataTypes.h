#pragma once

#include <vector_types.h>
#include "GenericCameraDataTypes.h"

// Define the perspective camera record
struct PerspectiveCameraData
{
  float3 scale;
  // x, y -> Aspect
  // z -> focal length/FOV

  inline bool operator==(const PerspectiveCameraData& other)
  { return (this->scale.x == other.scale.x && this->scale.y == other.scale.y && this->scale.z == other.scale.z); }
};

// A typedef for a RaygenPosedContainer containing a PerspectiveCameraData:
typedef cray::RaygenPosedContainer<PerspectiveCameraData> PerspectiveCameraPosedData;
