#include <glad/glad.h> 

#include "SimulationGPU.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_ERROR() checkLastError(__FILE__, __LINE__)

void checkLastError(const char* const file, const int line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

constexpr int BLOCK_SIZE = 512;
__constant__ SimulationParams c_params;

__global__ void initBoidsKernel(float* pX, float* pY, float* pZ, float* vX, float* vY, float* vZ, unsigned long long seed) 
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_params.numBoids) return;

    curandState state;
    curand_init(seed, i, 0, &state);

    pX[i] = curand_uniform(&state) * c_params.width;
    pY[i] = curand_uniform(&state) * c_params.height;
    pZ[i] = curand_uniform(&state) * c_params.depth;

    float z = (curand_uniform(&state) * 2.0f) - 1.0f;

    float angle = curand_uniform(&state) * 2.0f * M_PI;

    float r = sqrtf(1.0f - z * z);

    float x = r * cosf(angle);
    float y = r * sinf(angle);


    float speed = c_params.minSpeed + curand_uniform(&state) * (c_params.maxSpeed - c_params.minSpeed);

    vX[i] = x * speed;
    vY[i] = y * speed;
    vZ[i] = z * speed;
}

__global__ void assignGridKernel(const float* __restrict__ pX, const float* __restrict__ pY, const float* __restrict__ pZ, int* hash, int* index) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_params.numBoids) return;

    index[i] = i;

    int cellX = (int)(pX[i] / c_params.cellSize);
    int cellY = (int)(pY[i] / c_params.cellSize);
    int cellZ = (int)(pZ[i] / c_params.cellSize);

    if(cellX < 0) cellX = 0;
    if (cellX >= c_params.gridWidth) cellX = c_params.gridWidth - 1;

    if (cellY < 0) cellY = 0;
    if (cellY >= c_params.gridHeight) cellY = c_params.gridHeight - 1;

    if (cellZ < 0) cellZ = 0;
    if (cellZ >= c_params.gridDepth) cellZ = c_params.gridDepth - 1;

    hash[i] = (cellZ * c_params.gridHeight * c_params.gridWidth) + (cellY * c_params.gridWidth) + cellX;
}

__global__ void reorderDataKernel(const int* __restrict__ gridIndex,
    const float* __restrict__ oldX, const float* __restrict__ oldY, const float* __restrict__ oldZ,
    const float* __restrict__ oldVX, const float* __restrict__ oldVY, const float* __restrict__ oldVZ,
    float* sortedX, float* sortedY, float* sortedZ, 
    float* sortedVX, float* sortedVY, float* sortedVZ) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_params.numBoids) return;

    int originalInd = gridIndex[i];

    sortedX[i] = oldX[originalInd];
    sortedY[i] = oldY[originalInd];
    sortedZ[i] = oldZ[originalInd];
    sortedVX[i] = oldVX[originalInd];
    sortedVY[i] = oldVY[originalInd];
    sortedVZ[i] = oldVZ[originalInd];
}

__global__ void findStartEndKernel(const int* __restrict__ gridHash, int* cellStart, int* cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_params.numBoids) return;

    int myHash = gridHash[i];

    if (i == 0 || myHash != gridHash[i - 1]) {
        cellStart[myHash] = i;
    }

    if (i == c_params.numBoids - 1 || myHash != gridHash[i + 1]) {
        cellEnd[myHash] = i + 1;
    }
}
__device__ void applyTurnFactor(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z) {
    if (pos_x < c_params.margin) vel_x += c_params.turnFactor;
    if (pos_x > c_params.width - c_params.margin) vel_x -= c_params.turnFactor;
    if (pos_y < c_params.margin) vel_y += c_params.turnFactor;
    if (pos_y > c_params.height - c_params.margin) vel_y -= c_params.turnFactor;
    if (pos_z < c_params.margin) vel_z += c_params.turnFactor;
    if (pos_z > c_params.depth - c_params.margin) vel_z -= c_params.turnFactor;
}
__device__ void applyRepulsion(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z)
{
    if (!c_params.isRepulsorActive) return;

    const float beamRadius = 150.0f;
    const float repulsionStrength = 200.0f;

    float toBoidX = pos_x - c_params.origin_x;
    float toBoidY = pos_y - c_params.origin_y;
    float toBoidZ = pos_z - c_params.origin_z;

    float t = toBoidX * c_params.dir_x + toBoidY * c_params.dir_y + toBoidZ * c_params.dir_z;

    if (t > 0.0f)
    {
        float closeX = c_params.origin_x + t * c_params.dir_x;
        float closeY = c_params.origin_y + t * c_params.dir_y;
        float closeZ = c_params.origin_z + t * c_params.dir_z;

        float perpX = pos_x - closeX;
        float perpY = pos_y - closeY;
        float perpZ = pos_z - closeZ;

        float distSq = perpX * perpX + perpY * perpY + perpZ * perpZ;

        if (distSq < beamRadius * beamRadius)
        {
            float dist = sqrtf(distSq);
            if (dist > 0.001f)
            {
                float falloff = 1.0f - (dist / beamRadius);
                float strength = (repulsionStrength * falloff) / dist;

                vel_x += perpX * strength;
                vel_y += perpY * strength;
                vel_z += perpZ * strength;
            }
        }
    }
}
__device__ void applySpeedLimits(float& vel_x, float& vel_y, float& vel_z)
{
    float speedSq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;
    if (speedSq > c_params.maxSpeed * c_params.maxSpeed) {
        float s = sqrtf(speedSq);
        vel_x = (vel_x / s) * c_params.maxSpeed;
        vel_y = (vel_y / s) * c_params.maxSpeed;
        vel_z = (vel_z / s) * c_params.maxSpeed;
    }
    else if (speedSq < c_params.minSpeed * c_params.minSpeed) {
        float s = sqrtf(speedSq);
        if (s > 0.0001f) {
            vel_x = (vel_x / s) * c_params.minSpeed;
            vel_y = (vel_y / s) * c_params.minSpeed;
            vel_z = (vel_z / s) * c_params.minSpeed;
        }
    }
}

__global__ void updateGridBoidsKernel(const float* __restrict__ pX, const float* __restrict__ pY, const float* __restrict__ pZ,
    const float* __restrict__ vX, const float* __restrict__ vY, const float* __restrict__ vZ,
    const int* __restrict__ cellStart, const int* __restrict__ cellEnd,
    float* outPx, float* outPy, float* outPz,
    float* outVx, float* outVy, float* outVz,
    float dt)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_params.numBoids) return;

    float pos_x = pX[i];
    float pos_y = pY[i];
    float pos_z = pZ[i];

    float vel_x = vX[i];
    float vel_y = vY[i];
    float vel_z = vZ[i];

    float close_dx = 0.0f, close_dy = 0.0f, close_dz = 0.0f;
    float pos_x_avg = 0.0f, pos_y_avg = 0.0f, pos_z_avg = 0.0f;
    float vel_x_avg = 0.0f, vel_y_avg = 0.0f, vel_z_avg = 0.0f;
    float neighbors_count = 0.0f;

    int grid_x = (int)(pos_x / c_params.cellSize);
    int grid_y = (int)(pos_y / c_params.cellSize);
    int grid_z = (int)(pos_z / c_params.cellSize);
    for (int z = -1; z <= 1; z++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                int neighbor_x = grid_x + x;
                int neighbor_y = grid_y + y;
                int neighbor_z = grid_z + z;

                if (neighbor_x < 0 || neighbor_x >= c_params.gridWidth ||
                    neighbor_y < 0 || neighbor_y >= c_params.gridHeight ||
                    neighbor_z < 0 || neighbor_z >= c_params.gridDepth) continue;

                int cell_id = (neighbor_z * c_params.gridHeight * c_params.gridWidth) + (neighbor_y * c_params.gridWidth) + neighbor_x;
                int start_id = cellStart[cell_id];

                if (start_id == -1) continue;
                int end_id = cellEnd[cell_id];

                for (int k = start_id; k < end_id; k++)
                {
                    if (i == k) continue;

                    float other_px = pX[k];
                    float other_py = pY[k];
                    float other_pz = pZ[k];

                    float dx = pos_x - other_px;
                    float dy = pos_y - other_py;
                    float dz = pos_z - other_pz;

                    float distSq = dx * dx + dy * dy + dz * dz;

                    if (distSq < c_params.visualRangeSq)
                    {
                        if (distSq < c_params.protectedRangeSq)
                        {
                            close_dx += dx;
                            close_dy += dy;
                            close_dz += dz;
                        }
                        else
                        {
                            vel_x_avg += vX[k];
                            vel_y_avg += vY[k];
                            vel_z_avg += vZ[k];

                            pos_x_avg += other_px;
                            pos_y_avg += other_py;
                            pos_z_avg += other_pz;

                            neighbors_count += 1.0f;
                        }
                    }
                }
            }
        }
    }

    if(neighbors_count > 0.0f) {
        vel_x_avg /= neighbors_count;
        vel_y_avg /= neighbors_count;
        vel_z_avg /= neighbors_count;

        pos_x_avg /= neighbors_count;
        pos_y_avg /= neighbors_count;
        pos_z_avg /= neighbors_count;

        vel_x += (vel_x_avg - vel_x) * c_params.matchingFactor;
        vel_y += (vel_y_avg - vel_y) * c_params.matchingFactor;
        vel_z += (vel_z_avg - vel_z) * c_params.matchingFactor;

        vel_x += (pos_x_avg - pos_x) * c_params.centeringFactor;
        vel_y += (pos_y_avg - pos_y) * c_params.centeringFactor;
        vel_z += (pos_z_avg - pos_z) * c_params.centeringFactor;

    }

    vel_x += close_dx * c_params.avoidFactor;
    vel_y += close_dy * c_params.avoidFactor;
    vel_z += close_dz * c_params.avoidFactor;
    
    applyTurnFactor(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z);
    applySpeedLimits(vel_x, vel_y, vel_z);
    applyRepulsion(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z);

    outPx[i] = pos_x + vel_x * dt;
    outPy[i] = pos_y + vel_y * dt;
    outPz[i] = pos_z + vel_z * dt;

    outVx[i] = vel_x;
    outVy[i] = vel_y;
    outVz[i] = vel_z;
}


SimulationGPU::SimulationGPU(int numBoids, float width, float height, float depth)
{
    m_hostParams.numBoids = numBoids;

    m_hostParams.width = width;
    m_hostParams.height = height;
    m_hostParams.depth = depth;
    m_hostParams.margin = 100.0f;

    m_hostParams.visualRange = 40.0f;
    m_hostParams.protectedRange = 8.0f;

    m_hostParams.visualRangeSq = m_hostParams.visualRange * m_hostParams.visualRange;
    m_hostParams.protectedRangeSq = m_hostParams.protectedRange * m_hostParams.protectedRange;

    m_hostParams.maxSpeed = 25.0f;
    m_hostParams.minSpeed = 10.0f;

    m_hostParams.centeringFactor = 0.005f;
    m_hostParams.avoidFactor = 0.05f;
    m_hostParams.matchingFactor = 0.1f;
    m_hostParams.turnFactor = 5.0f;

    m_hostParams.cellSize = m_hostParams.visualRange;
    m_hostParams.gridWidth = (int)(ceil(width / m_hostParams.cellSize));
    m_hostParams.gridHeight = (int)(ceil(height / m_hostParams.cellSize));
    m_hostParams.gridDepth = (int)(ceil(depth / m_hostParams.cellSize));

    m_hostParams.numCells = m_hostParams.gridWidth * m_hostParams.gridHeight * m_hostParams.gridDepth;

    size_t sizeFloat = numBoids * sizeof(float);

    HANDLE_ERROR(cudaMalloc(&d_posX, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_posY, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_posZ, sizeFloat));

    HANDLE_ERROR(cudaMalloc(&d_velX, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_velY, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_velZ, sizeFloat));

    HANDLE_ERROR(cudaMalloc(&d_sortedPosX, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_sortedPosY, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_sortedPosZ, sizeFloat));

    HANDLE_ERROR(cudaMalloc(&d_sortedVelX, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_sortedVelY, sizeFloat));
    HANDLE_ERROR(cudaMalloc(&d_sortedVelZ, sizeFloat));


    size_t sizeInt = numBoids * sizeof(int);
    HANDLE_ERROR(cudaMalloc(&d_gridHash, sizeInt));
    HANDLE_ERROR(cudaMalloc(&d_gridIndex, sizeInt));

    size_t sizeCells = m_hostParams.numCells * sizeof(int);
    HANDLE_ERROR(cudaMalloc(&d_cellStart, sizeCells));
    HANDLE_ERROR(cudaMalloc(&d_cellEnd, sizeCells));


    updateParamsOnGPU();

    initRandomBoidsOnGPU();

}

void SimulationGPU::initRandomBoidsOnGPU()
{
    int blocks = (m_hostParams.numBoids + BLOCK_SIZE - 1) / BLOCK_SIZE;

    initBoidsKernel << <blocks, BLOCK_SIZE >> > (d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, time(NULL));
    CHECK_LAST_ERROR();

    cudaDeviceSynchronize();

}

void SimulationGPU::assignGridOnGPU()
{
    int blocks = (m_hostParams.numBoids + BLOCK_SIZE - 1) / BLOCK_SIZE;

    assignGridKernel << <blocks, BLOCK_SIZE >> > (d_posX, d_posY, d_posZ, d_gridHash, d_gridIndex);
    CHECK_LAST_ERROR();

    cudaDeviceSynchronize();
}

void SimulationGPU::reorderDataOnGPU()
{
    int blocks = (m_hostParams.numBoids + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    reorderDataKernel << <blocks, BLOCK_SIZE >> > (d_gridIndex, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, d_sortedPosX, d_sortedPosY, d_sortedPosZ, d_sortedVelX, d_sortedVelY, d_sortedVelZ);
    CHECK_LAST_ERROR();

    cudaDeviceSynchronize();
}

void SimulationGPU::findStartEndOnGPU()
{
    int blocks = (m_hostParams.numCells + BLOCK_SIZE - 1) / BLOCK_SIZE;

    HANDLE_ERROR(cudaMemset(d_cellStart, 0xFF, m_hostParams.numCells * sizeof(int)));

    findStartEndKernel << <blocks, BLOCK_SIZE >> > (d_gridHash, d_cellStart, d_cellEnd);
    CHECK_LAST_ERROR();

    cudaDeviceSynchronize();
}


void SimulationGPU::update(float deltaTime) {
    
    assignGridOnGPU();

    thrust::device_ptr<int> t_key(d_gridHash);
    thrust::device_ptr<int> t_val(d_gridIndex);

    thrust::sort_by_key(t_key, t_key + m_hostParams.numBoids, t_val);

    reorderDataOnGPU();
    findStartEndOnGPU();

    int blocks = (m_hostParams.numBoids + BLOCK_SIZE - 1) / BLOCK_SIZE;

    updateGridBoidsKernel << <blocks, BLOCK_SIZE >> > (
        d_sortedPosX, d_sortedPosY, d_sortedPosZ,
        d_sortedVelX, d_sortedVelY, d_sortedVelZ,
        d_cellStart, d_cellEnd,
        d_posX, d_posY, d_posZ,
        d_velX, d_velY, d_velZ,
        deltaTime
        );
    CHECK_LAST_ERROR();

    if (res_vboPosX)
    {
        HANDLE_ERROR(cudaGraphicsMapResources(1, &res_vboPosX, 0));
        HANDLE_ERROR(cudaGraphicsMapResources(1, &res_vboPosY, 0));
        HANDLE_ERROR(cudaGraphicsMapResources(1, &res_vboPosZ, 0));

        HANDLE_ERROR(cudaGraphicsMapResources(1, &res_vboVelX, 0));
        HANDLE_ERROR(cudaGraphicsMapResources(1, &res_vboVelY, 0));
        HANDLE_ERROR(cudaGraphicsMapResources(1, &res_vboVelZ, 0));

        float* ptrX, * ptrY, * ptrZ, * ptrVX, * ptrVY, * ptrVZ;
        size_t s;
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&ptrX, &s, res_vboPosX));
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&ptrY, &s, res_vboPosY));
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&ptrZ, &s, res_vboPosZ));

        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&ptrVX, &s, res_vboVelX));
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&ptrVY, &s, res_vboVelY));
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&ptrVZ, &s, res_vboVelZ));

        size_t bytes = m_hostParams.numBoids * sizeof(float);
        HANDLE_ERROR(cudaMemcpy(ptrX, d_posX, bytes, cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(ptrY, d_posY, bytes, cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(ptrZ, d_posZ, bytes, cudaMemcpyDeviceToDevice));

        HANDLE_ERROR(cudaMemcpy(ptrVX, d_velX, bytes, cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(ptrVY, d_velY, bytes, cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(ptrVZ, d_velZ, bytes, cudaMemcpyDeviceToDevice));

        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &res_vboPosX, 0));
        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &res_vboPosY, 0));
        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &res_vboPosZ, 0));

        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &res_vboVelX, 0));
        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &res_vboVelY, 0));
        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &res_vboVelZ, 0));

    }
}

void SimulationGPU::updateParamsOnGPU() {
    HANDLE_ERROR(cudaMemcpyToSymbol(c_params, &m_hostParams, sizeof(SimulationParams)));
}

void SimulationGPU::setInteractionData(bool active, float origin_x, float origin_y, float origin_z, float dir_x, float dir_y, float dir_z)
{
    m_hostParams.isRepulsorActive = active ? 1 : 0;

    m_hostParams.origin_x = origin_x;
    m_hostParams.origin_y = origin_y;
    m_hostParams.origin_z = origin_z;

    m_hostParams.dir_x = dir_x;
    m_hostParams.dir_y = dir_y;
    m_hostParams.dir_z = dir_z;

    updateParamsOnGPU();
}

void SimulationGPU::registerVBOs(unsigned int VBO_PosX, unsigned int VBO_PosY, unsigned int VBO_PosZ, unsigned int VBO_VelX, unsigned int VBO_VelY, unsigned int VBO_VelZ)
{
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&res_vboPosX, VBO_PosX, cudaGraphicsMapFlagsWriteDiscard));
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&res_vboPosY, VBO_PosY, cudaGraphicsMapFlagsWriteDiscard));
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&res_vboPosZ, VBO_PosZ, cudaGraphicsMapFlagsWriteDiscard));

    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&res_vboVelX, VBO_VelX, cudaGraphicsMapFlagsWriteDiscard));
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&res_vboVelY, VBO_VelY, cudaGraphicsMapFlagsWriteDiscard));
    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&res_vboVelZ, VBO_VelZ, cudaGraphicsMapFlagsWriteDiscard));
}
void SimulationGPU::unregisterVBOs()
{
    if (res_vboPosX) cudaGraphicsUnregisterResource(res_vboPosX);
    if (res_vboPosY) cudaGraphicsUnregisterResource(res_vboPosY);
    if (res_vboPosZ) cudaGraphicsUnregisterResource(res_vboPosZ);

    if (res_vboVelX) cudaGraphicsUnregisterResource(res_vboVelX);
    if (res_vboVelY) cudaGraphicsUnregisterResource(res_vboVelY);
    if (res_vboVelZ) cudaGraphicsUnregisterResource(res_vboVelZ);

    res_vboPosX = nullptr;
    res_vboPosY = nullptr;
    res_vboPosZ = nullptr;

    res_vboVelX = nullptr;
    res_vboVelY = nullptr;
    res_vboVelZ = nullptr;
}

SimulationGPU::~SimulationGPU()
{
    unregisterVBOs();

    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);

    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velZ);

    cudaFree(d_sortedPosX);
    cudaFree(d_sortedPosY);
    cudaFree(d_sortedPosZ);

    cudaFree(d_sortedVelX);
    cudaFree(d_sortedVelY);
    cudaFree(d_sortedVelZ);

    cudaFree(d_gridHash);
    cudaFree(d_gridIndex);

    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
}