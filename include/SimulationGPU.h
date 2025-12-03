#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "ISimulation.h"

struct SimulationParams {
    int numBoids;

    float width;
    float height;
    float depth;
    float margin;

    float visualRange;
    float protectedRange;
    float predatorDetectionMultiplier;

    float visualRangeSq;
    float protectedRangeSq;

    float maxSpeed;
    float minSpeed;

    float sharkMaxSpeed;

    float centeringFactor;
    float avoidFactor;
    float matchingFactor;
    float turnFactor;

    float cellSize;
    int gridWidth;
    int gridHeight;
    int gridDepth;
    int numCells;

    float repulsionStrength;
    int isRepulsorActive;
    float origin_x;
    float origin_y;
    float origin_z;
    float dir_x;
    float dir_y;
    float dir_z;
};


class SimulationGPU : public ISimulation
{
public:
    SimulationGPU(int numBoids, float width, float height, float depth);
    ~SimulationGPU();

    void update(float deltaTime) override;

    SimulationParams& getParamsRef() { return m_hostParams; }
    void updateParamsOnGPU();
    void setInteractionData(bool active, float origin_x, float origin_y, float origin_z, float dir_x, float dir_y, float dir_z) override;

    void registerVBOs(unsigned int VBO_PosX, unsigned int VBO_PosY, unsigned int VBO_PosZ, unsigned int VBO_VelX, unsigned int VBO_VelY, unsigned int VBO_VelZ, unsigned int VBO_type);
    void unregisterVBOs();

    SimulationStats getStats() const override { return m_stats; }
private:

    void applyTurnFactor(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z);
    void applySpeedLimits(float& vel_x, float& vel_y, float& vel_z);
    void applyRepulsion(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z);
    

    float *d_sortedPosX, *d_sortedPosY, *d_sortedPosZ;
    float *d_sortedVelX, *d_sortedVelY, *d_sortedVelZ;
    int* d_sorted_type;

    int* d_gridHash; 
    int* d_gridIndex;

    int* d_cellStart;
    int* d_cellEnd;


    cudaGraphicsResource* res_vboPosX = nullptr;
    cudaGraphicsResource* res_vboPosY = nullptr;
    cudaGraphicsResource* res_vboPosZ = nullptr;

    cudaGraphicsResource* res_vboVelX = nullptr;
    cudaGraphicsResource* res_vboVelY = nullptr;
    cudaGraphicsResource* res_vboVelZ = nullptr;

    cudaGraphicsResource* res_vboType = nullptr;

    SimulationParams m_hostParams;

    SimulationStats m_stats;

    cudaEvent_t startEvent, stopEvent;
};