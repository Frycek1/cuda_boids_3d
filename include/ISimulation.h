#pragma once

struct SimulationStats {
    float timeInit = 0.0f;       // Time to generate date (ms)
    float timeCopy = 0.0f;       // Time to copy map data (ms)
    float timeSort = 0.0f;       // Time to create and sort grid (ms)
    float timeUpdate = 0.0f;     // Time to calculate boids pos (ms)
    float timeTotal = 0.0f;      // Total simulation step time (ms)
};
class ISimulation
{
public:

    virtual ~ISimulation() = default;


    virtual void update(float deltaTime) = 0;
    virtual void setInteractionData(bool active, float origin_x, float origin_y, float origin_z, float dir_x, float dir_y, float dir_z) = 0;
    virtual SimulationStats getStats() const = 0;
};
