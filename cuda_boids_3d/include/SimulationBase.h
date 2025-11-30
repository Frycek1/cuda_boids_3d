#pragma once
#include <vector>
#include <glm/glm.hpp>        
#include <glm/gtc/random.hpp>

enum BoidGroup { NONE, SCOUT_GROUP_1, SCOUT_GROUP_2 };

class SimulationBase
{
public:
    // 3D Vectors
    std::vector<float> positions_x;
    std::vector<float> positions_y;
    std::vector<float> positions_z; // <-- NOWE

    std::vector<float> velocities_x;
    std::vector<float> velocities_y;
    std::vector<float> velocities_z; // <-- NOWE

    std::vector<BoidGroup> groups;
    std::vector<float> bias_vals;

    int numBoids;

    // Parameters
    float visualRange, protectedRange;
    float centeringfactor, avoidfactor, matchingfactor, turnfactor;
    float maxspeed, minspeed;
    float visualRangeSq, protectedRangeSq;

    // Wymiary świata
    float screenWidth, screenHeight, screenDepth; // <-- NOWE
    float margin;

public:
    SimulationBase(int numBoids, int numScout1, int numScout2, float width, float height);
    virtual ~SimulationBase() {}

    // Interfejs
    virtual void update(float deltaTime) = 0;
    int getNumBoids() const { return numBoids; }

    // Gettery 3D
    const std::vector<float>& getPosX() const { return positions_x; }
    const std::vector<float>& getPosY() const { return positions_y; }
    const std::vector<float>& getPosZ() const { return positions_z; }

    const std::vector<float>& getVelX() const { return velocities_x; }
    const std::vector<float>& getVelY() const { return velocities_y; }
    const std::vector<float>& getVelZ() const { return velocities_z; }

protected:
    void enforceSpeedLimits(int i);
    void applyScreenEdges(int i);
    void applyBias(int i);

private:
    void createNewBoid(BoidGroup group, float biasval);
};