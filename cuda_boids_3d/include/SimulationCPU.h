#pragma once
#include "ISimulation.h"
#include <vector>
#include <glm/glm.hpp>        
#include <glm/gtc/random.hpp>

class SimulationCPU : public ISimulation
{
public:
    std::vector<float> positions_x;
    std::vector<float> positions_y;
    std::vector<float> positions_z; 

    std::vector<float> velocities_x;
    std::vector<float> velocities_y;
    std::vector<float> velocities_z; 

    int numBoids;

    float visualRange, protectedRange;
    float visualRangeSq, protectedRangeSq;

    float centeringfactor, avoidfactor, matchingfactor, turnfactor;
    float maxspeed, minspeed;

    float screenWidth, screenHeight, screenDepth;
    float margin;
public:
    SimulationCPU(int numBoids, float width, float height, float depth);
    void update(float deltaTime) override;

    int getNumBoids() const { return numBoids; }

    const std::vector<float>& getPosX() const { return positions_x; }
    const std::vector<float>& getPosY() const { return positions_y; }
    const std::vector<float>& getPosZ() const { return positions_z; }

    const std::vector<float>& getVelX() const { return velocities_x; }
    const std::vector<float>& getVelY() const { return velocities_y; }
    const std::vector<float>& getVelZ() const { return velocities_z; }
private:
    // Grid structures
    std::vector<int> boid_cellIDs;
    std::vector<int> boid_indices;
    std::vector<int> cell_start_indices;

    // Next buffers (Double Buffering)
    std::vector<float> n_posX, n_posY, n_posZ;
    std::vector<float> n_velX, n_velY, n_velZ;

    float cellSize;
    int gridWidth, gridHeight, gridDepth; // <-- 3D Grid
    int numCells;

    // Pomocnicza
    int getCellID(float x, float y, float z) const;

    void enforceSpeedLimits(int i);
    void applyScreenEdges(int i);
    void applyBias(int i);

    void createNewBoid();
};