#pragma once

class ISimulation
{
public:

    virtual ~ISimulation() = default;


    virtual void update(float deltaTime) = 0;
    virtual void setInteractionData(bool active, float origin_x, float origin_y, float origin_z, float dir_x, float dir_y, float dir_z) = 0;
};
