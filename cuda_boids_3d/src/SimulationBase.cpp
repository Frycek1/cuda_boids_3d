#include "SimulationBase.h"
#include <cmath>
#include <algorithm>

using namespace glm;

SimulationBase::SimulationBase(int numBoids, int numScout1, int numScout2, float width, float height)
    : screenWidth(width), screenHeight(height)
{
    this->numBoids = numBoids;

    // Ustawiamy głębokość taką samą jak wysokość (sześcian)
    this->screenDepth = height;

    visualRange = 40.0f;
    protectedRange = 8.0f;
    centeringfactor = 0.0005f;
    avoidfactor = 0.05f;
    matchingfactor = 0.05f;
    turnfactor = 0.5f; // Zwiększamy, żeby szybciej odbijały
    margin = 100.0f;
    maxspeed = 6.0f;
    minspeed = 3.0f;
    visualRangeSq = visualRange * visualRange;
    protectedRangeSq = protectedRange * protectedRange;

    // Rezerwacja pamięci
    positions_x.reserve(numBoids); positions_y.reserve(numBoids); positions_z.reserve(numBoids);
    velocities_x.reserve(numBoids); velocities_y.reserve(numBoids); velocities_z.reserve(numBoids);
    groups.reserve(numBoids);
    bias_vals.reserve(numBoids);

    // Tworzenie boidów
    int created = 0;
    while (created < numBoids) {
        createNewBoid(NONE, 0.001f);
        created++;
    }
}

void SimulationBase::createNewBoid(BoidGroup group, float biasval)
{
    // Losowy wektor na sferze
    vec3 randDir = glm::sphericalRand(1.0f); // Z glm/gtc/random.hpp
    float speed = (minspeed + maxspeed) * 0.5f;

    positions_x.push_back(glm::linearRand(0.0f, screenWidth));
    positions_y.push_back(glm::linearRand(0.0f, screenHeight));
    positions_z.push_back(glm::linearRand(0.0f, screenDepth)); // <-- Z

    velocities_x.push_back(randDir.x * speed);
    velocities_y.push_back(randDir.y * speed);
    velocities_z.push_back(randDir.z * speed); // <-- Z

    groups.push_back(group);
    bias_vals.push_back(biasval);
}

void SimulationBase::applyScreenEdges(int i)
{
    // X
    if (positions_x[i] < margin) velocities_x[i] += turnfactor;
    if (positions_x[i] > screenWidth - margin) velocities_x[i] -= turnfactor;
    // Y
    if (positions_y[i] < margin) velocities_y[i] += turnfactor;
    if (positions_y[i] > screenHeight - margin) velocities_y[i] -= turnfactor;
    // Z (Głębokość)
    if (positions_z[i] < margin) velocities_z[i] += turnfactor;
    if (positions_z[i] > screenDepth - margin) velocities_z[i] -= turnfactor;
}

void SimulationBase::applyBias(int i)
{
    // (Opcjonalne - bez zmian, tylko dodaj obsługę Z jeśli chcesz bias w 3D)
}

void SimulationBase::enforceSpeedLimits(int i)
{
    float vx = velocities_x[i];
    float vy = velocities_y[i];
    float vz = velocities_z[i];

    float speedSq = vx * vx + vy * vy + vz * vz;

    if (speedSq > maxspeed * maxspeed) {
        float s = std::sqrt(speedSq);
        velocities_x[i] = (vx / s) * maxspeed;
        velocities_y[i] = (vy / s) * maxspeed;
        velocities_z[i] = (vz / s) * maxspeed;
    }
    else if (speedSq < minspeed * minspeed) {
        float s = std::sqrt(speedSq);
        if (s > 0.00001f) {
            velocities_x[i] = (vx / s) * minspeed;
            velocities_y[i] = (vy / s) * minspeed;
            velocities_z[i] = (vz / s) * minspeed;
        }
    }
}