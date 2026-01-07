#include "CompoundEye.h"
#include "curand_kernel.h"

RaygenRecord<RecordPointer> CompoundEye::s_compoundRecordPtrRecord = (RaygenRecord<RecordPointer>){};
CUdeviceptr CompoundEye::s_d_compoundRecordPtrRecord = (CUdeviceptr){};

CompoundEye::CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount, const std::string& eyeDataPath) : DataRecordCamera<CompoundEyeData>(name), shaderName(NAME_PREFIX + shaderName)
{
    //// Assign VRAM for compound eye structure configuration
    reconfigureOmmatidialCount(ommatidialCount);

    // Set this object's eyeDataPath to a copy of the given eyeDataPath
    this->eyeDataPath = std::string(eyeDataPath);

}
CompoundEye::~CompoundEye()
{
    // Free VRAM of compound eye structure information
    freeOmmatidialMemory();
    // Free VRAM of the compound eye random states
    freeOmmatidialRandomStates();
    // Free VRAM of the compound eye's rendering buffer
}

void CompoundEye::setShaderName(const std::string shaderName)
{
    this->shaderName = NAME_PREFIX + shaderName;
}

void CompoundEye::setOmmatidia(Ommatidium* ommatidia, size_t count)
{
    reconfigureOmmatidialCount(count); // Change the count and buffers (if required)
    copyOmmatidia(ommatidia); // Actually copy the data in
}
void CompoundEye::reconfigureOmmatidialCount(size_t count)
{
    // Only do this if the count has changed
    if(count != specializedData.ommatidialCount)
    {
        // Assign VRAM for compound eye structure configuration
        specializedData.ommatidialCount = count;
        allocateOmmatidialMemory();
        // Assign VRAM for the random states
        allocateOmmatidialRandomStates();
        // Assign VRAM for the compound rendering buffer
        allocateCompoundRenderingBuffer();
        allocateCompoundRenderingAvgBuffer();
    }
}

// Copies the averages in specializedData.d_compoundAvgBuffer to this->ommatidial_average
void CompoundEye::copyOmmatidialDataToHost()
{
    CUDA_CHECK( cudaMemcpy(this->ommatidial_average, // destination
                           reinterpret_cast<void*>(specializedData.d_compoundAvgBuffer), // source
                           sizeof(float3) * specializedData.ommatidialCount,
                           cudaMemcpyDeviceToHost) );
    CUDA_SYNC_CHECK();
}

// This should be called after averageRecordFrame() has been executed
float3* CompoundEye::getRecordFrame()
{
    this->copyOmmatidialDataToHost(); // Copies the data in specializedData.d_compoundAvgBuffer
    return this->ommatidial_average;
}

#include "summing_kernel.h"
void CompoundEye::averageRecordFrame()
{
    uint32_t omc = this->getOmmatidialCount();
    uint32_t spo = this->getSamplesPerOmmatidium();
    // This launches a CUDA kernel to do the reduction of all the samples in
    // d_compoundBuffer down to the averages, which end up in d_compoundAvgBuffer
    summing_kernel (reinterpret_cast<float3*>(specializedData.d_compoundBuffer),
                    reinterpret_cast<float3*>(specializedData.d_compoundAvgBuffer), omc, spo);
}

void CompoundEye::zeroRecordFrame()
{
    CUDA_CHECK( cudaMemset(reinterpret_cast<void*>(specializedData.d_compoundAvgBuffer), 0, sizeof(float3) * specializedData.ommatidialCount) );
    CUDA_SYNC_CHECK();
}

void CompoundEye::copyOmmatidia(Ommatidium* ommatidia)
{
    CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(specializedData.d_ommatidialArray),
                    ommatidia,
                    sizeof(Ommatidium)*specializedData.ommatidialCount,
                    cudaMemcpyHostToDevice
                    )
        );
    CUDA_SYNC_CHECK();
}

void CompoundEye::allocateOmmatidialMemory()
{
    size_t memSize = sizeof(Ommatidium)*specializedData.ommatidialCount;
    if constexpr (debug_memory == true) {
        std::cout << "Clearing and allocating ommatidial data on device. "
                  << "(size: "<<memSize<<", "<<specializedData.ommatidialCount<<" blocks)"<<std::endl;
    }
    freeOmmatidialMemory();
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_ommatidialArray) ), memSize) );

    if constexpr (debug_memory == true) {
        CUDA_CHECK( cudaMemset(
                        reinterpret_cast<void*>(specializedData.d_ommatidialArray),
                        0,
                        memSize
                        ) );
    }

    if constexpr (debug_memory == true) { std::cout << "\t...allocated at " << specializedData.d_ommatidialArray << std::endl; }
    CUDA_SYNC_CHECK();

    // FIXME: malloc ommatidial_average when cudaMallocing the associated cuda buffer
    // Also allocate ommatidial_average, our final output buffer
    this->ommatidial_average = (float3*) malloc (sizeof(float3) * specializedData.ommatidialCount);
}
void CompoundEye::freeOmmatidialMemory()
{
    if constexpr (debug_memory == true) {
        std::cout << "[CAMERA: " << getCameraName() << "] Freeing ommatidial memory... ";
    }
    if(specializedData.d_ommatidialArray != 0)
    {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_ommatidialArray)) );
        specializedData.d_ommatidialArray = 0;
        if constexpr (debug_memory == true) { std::cout << "Ommatidial memory freed!" << std::endl; }
    } else{
        if constexpr (debug_memory == true) { std::cout << "Ommatidial memory already free, skipping..." << std::endl; }
    }
    CUDA_SYNC_CHECK();

    // Free our host-side average, too
    if (this->ommatidial_average != nullptr) {
        free (this->ommatidial_average);
        this->ommatidial_average = nullptr;
    }
}

void CompoundEye::allocateOmmatidialRandomStates()
{
    size_t blockCount = specializedData.ommatidialCount * specializedData.samplesPerOmmatidium;// The number of cuRand states
    size_t memSize = sizeof(curandState)*blockCount;
    if constexpr (debug_memory == true) {
        std::cout << "[CAMERA: " << getCameraName() << "] Clearing and allocating per-ommatidium random states on device. (size: "
                  << memSize << ", " << blockCount << " blocks)" << std::endl;
    }
    freeOmmatidialRandomStates();
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_randomStates) ), memSize) );

    if constexpr (debug_memory == true) {
        CUDA_CHECK( cudaMemset(
                        reinterpret_cast<void*>(specializedData.d_randomStates),
                        0,
                        memSize
                        ) );
    }


    if constexpr (debug_memory == true) { std::cout << "\t...allocated at " << specializedData.d_randomStates << std::endl; }
    // TODO(RANDOMS): The randomStateBuffer is currently unitialized. For now we'll be initializing it with if statements in the ommatidial shader, but in the future a CUDA function could be called here to initialize it.
    // Set this camera's randomsConfigured switch to false so that the aforementioned if statement can work:
    specializedData.randomsConfigured = false;
    CUDA_SYNC_CHECK();

}
void CompoundEye::freeOmmatidialRandomStates()
{
    if constexpr (debug_memory == true) {
        std::cout << "[CAMERA: " << getCameraName() << "] Freeing ommatidial random states... ";
    }
    if(specializedData.d_randomStates != 0)
    {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_randomStates)) );
        specializedData.d_randomStates = 0;
        if constexpr (debug_memory == true) { std::cout << "Ommatidial random states freed!" << std::endl; }
    } else {
        if constexpr (debug_memory == true) { std::cout << "Ommatidial random states already free, skipping..." << std::endl; }
    }
    CUDA_SYNC_CHECK();
}
void CompoundEye::allocateCompoundRenderingBuffer()
{
    size_t blockCount = specializedData.ommatidialCount * specializedData.samplesPerOmmatidium;
    size_t memSize = sizeof(float3)*blockCount;
    if constexpr (debug_memory == true) {
        std::cout << "[CAMERA: " << getCameraName() << "] Allocating compound render buffer (size: "
                  << sizeof(float3) << " x " << specializedData.ommatidialCount
                  << " x " << specializedData.samplesPerOmmatidium  << " (sizof x omcount x omsamples)" << std::endl;
        std::cout << "[CAMERA: " << getCameraName() << "] Clearing and allocating compound render buffer on device. "
                  << "(size: "<<memSize<<", "<<blockCount<<" blocks)"<<std::endl;
    }
    freeCompoundRenderingBuffer();
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_compoundBuffer) ), memSize) );

    if constexpr (debug_memory == true) {
        CUDA_CHECK( cudaMemset(
                        reinterpret_cast<void*>(specializedData.d_compoundBuffer),
                        0,
                        memSize
                        ) );
    }


    if constexpr (debug_memory == true) { std::cout << "...allocated at " << specializedData.d_compoundBuffer << std::endl; }
    CUDA_SYNC_CHECK();
}
void CompoundEye::freeCompoundRenderingBuffer()
{
    if constexpr (debug_memory == true) {
        std::cout << "[CAMERA: " << getCameraName() << "] Freeing compound render buffer... ";
    }
    if(specializedData.d_compoundBuffer != 0)
    {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_compoundBuffer)) );
        specializedData.d_compoundBuffer= 0;
        if constexpr (debug_memory == true) { std::cout << "freed!" << std::endl; }
    } else{
        if constexpr (debug_memory == true) { std::cout << "already free, skipping..." << std::endl; }
    }
    CUDA_SYNC_CHECK();
}

void CompoundEye::allocateCompoundRenderingAvgBuffer()
{
    size_t blockCount = specializedData.ommatidialCount;
    size_t memSize = sizeof(float3)*blockCount;
    if constexpr (debug_memory == true) {
        std::cout << "[CAMERA: " << getCameraName() << "] "
                  << "Allocating compound render AVG buffer "
                  << "(size: " << sizeof(float3) << " x " << specializedData.ommatidialCount
                  << " (sizeof x omcount)" << std::endl;
        std::cout << "[CAMERA: " << getCameraName() << "] "
                  << "Clearing and allocating compound render AVG buffer on device. "
                  << "(size: " << memSize << ", " << blockCount << " blocks)" << std::endl;
    }
    freeCompoundRenderingAvgBuffer();
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_compoundAvgBuffer) ), memSize) );

    if constexpr (debug_memory == true) {
        CUDA_CHECK( cudaMemset(
                        reinterpret_cast<void*>(specializedData.d_compoundAvgBuffer),
                        0,
                        memSize
                        ) );
    }

    if constexpr (debug_memory == true) { std::cout << "[CAMERA: " << getCameraName() << "] Compound AVG buffer re-allocated\n"; }
    CUDA_SYNC_CHECK();
}
void CompoundEye::freeCompoundRenderingAvgBuffer()
{
    if constexpr (debug_memory == true) {
        std::cout << "[CAMERA: " << getCameraName() << "] Freeing compound AVG render buffer... ";
    }
    if(specializedData.d_compoundAvgBuffer != 0)
    {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_compoundAvgBuffer)) );
        specializedData.d_compoundAvgBuffer = 0;
        if constexpr (debug_memory == true) { std::cout << "buffer freed!" << std::endl; }
    } else {
        if constexpr (debug_memory == true) { std::cout << "buffer already free, skipping." << std::endl; }
    }
    CUDA_SYNC_CHECK();
}

void CompoundEye::setSamplesPerOmmatidium(int32_t s)
{
    specializedData.samplesPerOmmatidium = max(static_cast<int32_t>(1),s);
    allocateOmmatidialRandomStates();
    allocateCompoundRenderingBuffer();
    allocateCompoundRenderingAvgBuffer();
    if constexpr (debug_memory == true) {
        std::cout << "Set samples per ommatidium to " << specializedData.samplesPerOmmatidium << std::endl;
    }
}
void CompoundEye::changeSamplesPerOmmatidiumBy(int32_t d)
{
    std::cout << "Changing samples per ommatidium from " << specializedData.samplesPerOmmatidium << " to "
              << (specializedData.samplesPerOmmatidium + d) << std::endl;
    setSamplesPerOmmatidium(specializedData.samplesPerOmmatidium + d);
}

// ----------------------------------------------------------------
//    Compound record handling
// ----------------------------------------------------------------

void CompoundEye::InitiateCompoundRecord(OptixShaderBindingTable& compoundSbt, OptixProgramGroup& compoundProgramGroup, const CUdeviceptr& targetRecord)
{
    // Allocate compound record (pointer to a camera) on device VRAM
    if constexpr (debug_memory == true) {
        std::cout << "Allocating compound SBT pointer record on device (size: " << sizeof(s_compoundRecordPtrRecord) << ")..." << std::endl;
    }
    if (s_d_compoundRecordPtrRecord != 0) {
        if constexpr (debug_memory == true) {
            std::cout << "\tWARN: Attempt to allocate compound SBT pointer record was made when one is already allocated." << std::endl;
        }
        return;
    }

    FreeCompoundRecord();

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&s_d_compoundRecordPtrRecord), sizeof(s_compoundRecordPtrRecord)) );

    if constexpr (debug_memory == true) {
        CUDA_CHECK( cudaMemset(
                        reinterpret_cast<void*>(s_d_compoundRecordPtrRecord),
                        0,
                        sizeof(s_compoundRecordPtrRecord)
                        ) );
    }


    if constexpr (debug_memory == true) { std::cout << "...allocated at " << s_d_compoundRecordPtrRecord << std::endl; }

    // Actually point the record to the target record
    // and update the VRAM to reflect this change
    // TODO: Replace the pointer below with a reference
    RedirectCompoundDataPointer(compoundProgramGroup, targetRecord);
    if constexpr (debug_memory == true) { std::cout << "Data redirected, setting record... "; }
    // Bind the record to the SBT
    compoundSbt.raygenRecord = s_d_compoundRecordPtrRecord;
}
void CompoundEye::FreeCompoundRecord()
{
    if constexpr (debug_memory == true) { std::cout << "Freeing compound SBT record... "; }
    if(s_d_compoundRecordPtrRecord != 0) {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(s_d_compoundRecordPtrRecord)) );
        s_d_compoundRecordPtrRecord = 0;
        if constexpr (debug_memory == true) { std::cout << "done!" << std::endl; }
    } else { if constexpr (debug_memory == true) { std::cout << "record already freed!" << std::endl; } }
}

void CompoundEye::RedirectCompoundDataPointer(OptixProgramGroup& programGroup, const CUdeviceptr& targetRecord)
{
    if constexpr (debug_memory == true) { std::cout << "Redirecting compound record pointer..." << std::endl; }
    s_compoundRecordPtrRecord.data.d_record = targetRecord;
    if constexpr (debug_memory == true) { std::cout << "\tPacking header..." << std::endl; }
    OPTIX_CHECK( optixSbtRecordPackHeader(programGroup, &s_compoundRecordPtrRecord) );
    if constexpr (debug_memory == true) { std::cout << "\tCopying to VRAM..."; }
    CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(s_d_compoundRecordPtrRecord),
                    &s_compoundRecordPtrRecord,
                    sizeof(s_compoundRecordPtrRecord),
                    cudaMemcpyHostToDevice
                    ) );
    if constexpr (debug_memory == true) {
        std::cout << "\t...Copy complete!\n\tCompound record redirected to " << targetRecord << std::endl;
    }
}
