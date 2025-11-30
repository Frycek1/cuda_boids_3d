#pragma once

class ISimulation
{
public:

    virtual ~ISimulation() = default;


    virtual void update(float deltaTime) = 0;

};