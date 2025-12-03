#pragma once
#include "ISimulation.h"
#include <vector>
#include <glm/glm.hpp>        
#include <glm/gtc/random.hpp>

class SimulationCPU : public ISimulation
{
public:
    std::vector<float> posX;
    std::vector<float> posY;
    std::vector<float> posZ; 

    std::vector<float> velX;
    std::vector<float> velY;
    std::vector<float> velZ; 

    std::vector<int> type;

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

    float repulsionStrength;
    bool isRepulsorActive = false;
    glm::vec3 origin = glm::vec3(0.0f);
    glm::vec3 dir = glm::vec3(0.0f);

    float globalTime = 0.0f;

public:
    SimulationCPU(int numBoids, float width, float height, float depth);
    void setInteractionData(bool active, float origin_x, float origin_y, float origin_z, float dir_x, float dir_y, float dir_z) override;

    void update(float deltaTime) override;

    int getNumBoids() const { return numBoids; }

    const std::vector<float>& getPosX() const { return posX; }
    const std::vector<float>& getPosY() const { return posY; }
    const std::vector<float>& getPosZ() const { return posZ; }

    const std::vector<float>& getVelX() const { return velX; }
    const std::vector<float>& getVelY() const { return velY; }
    const std::vector<float>& getVelZ() const { return velZ; }

    const std::vector<int>& getType() const { return type; }

    SimulationStats getStats() const override { return m_stats; }
private:
    std::vector<int> gridHash;
    std::vector<int> gridIndex;
    std::vector<int> cellStart;
    std::vector<int> cellEnd;

    std::vector<float> n_posX, n_posY, n_posZ;
    std::vector<float> n_velX, n_velY, n_velZ;

    float cellSize;
    int gridWidth;
    int gridHeight;
    int gridDepth;
    int numCells;

    SimulationStats m_stats;

    int getCellID(float x, float y, float z) const;

    void applyTurnFactor(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z);
    void applyRepulsion(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z);
    void applySpeedLimits(float& vel_x, float& vel_y, float& vel_z, float maxSpeed);

    void createNewBoid();
};