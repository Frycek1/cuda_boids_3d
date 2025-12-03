#include "SimulationCPU.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

using namespace glm;

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

float hash_cpu(float n) {
    return sinf(n) * 43758.5453123f - floorf(sinf(n) * 43758.5453123f);
}

glm::vec3 getWanderTarget_cpu(int id, float time, float frequency) {
    float t = floorf(time * frequency);

    float seed = (float)id + t * 13.57f;
    float u = hash_cpu(seed);
    float v = hash_cpu(seed + 123.45f);

    float theta = 2.0f * M_PI * u;
    float phi = acosf(2.0f * v - 1.0f);

    return glm::vec3(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
}

SimulationCPU::SimulationCPU(int numBoids, float width, float height, float depth) : width(width), height(height), depth(depth)
{
    auto start = std::chrono::high_resolution_clock::now();

    this->numBoids = numBoids;

    visualRange = 40.0f;
    protectedRange = 8.0f;
    predatorDetectionMultiplier = 1.0f;

    centeringFactor = 0.005f;
    avoidFactor = 0.05f;
    matchingFactor = 0.1f;
    turnFactor = 5.0f;
    margin = 100.0f;

    maxSpeed = 25.0f;
    minSpeed = 20.0f;
    sharkMaxSpeed = maxSpeed * 2.0f;

    visualRangeSq = visualRange * visualRange;
    protectedRangeSq = protectedRange * protectedRange;
    repulsionStrength = 200.0f;
    
    globalTime = 0.0f;

    posX.reserve(numBoids); 
    posY.reserve(numBoids); 
    posZ.reserve(numBoids);
    velX.reserve(numBoids); 
    velY.reserve(numBoids); 
    velZ.reserve(numBoids);

    for (int i = 0; i < numBoids; i++)
    {
        createNewBoid();
    }

    cellSize = visualRange;

    gridWidth = (int)std::ceil(width / cellSize);
    gridHeight = (int)std::ceil(height / cellSize);
    gridDepth = (int)std::ceil(depth / cellSize);

    numCells = gridWidth * gridHeight * gridDepth;

    gridHash.resize(numBoids);
    gridIndex.resize(numBoids);
    cellStart.resize(numCells);
    cellEnd.resize(numCells);

    n_posX.resize(numBoids); n_posY.resize(numBoids); n_posZ.resize(numBoids);
    n_velX.resize(numBoids); n_velY.resize(numBoids); n_velZ.resize(numBoids);

    auto end = std::chrono::high_resolution_clock::now();
    m_stats.timeInit = std::chrono::duration<float, std::milli>(end - start).count();
}

void SimulationCPU::createNewBoid()
{
    vec3 randDir = glm::sphericalRand(1.0f);

    float r = glm::linearRand(0.0f, 1.0f);
    bool isShark = (r < 0.001f);

    float speed;
    if (isShark) {
        speed = sharkMaxSpeed;
    }
    else {
        speed = glm::linearRand(minSpeed, maxSpeed);
    }

    posX.push_back(glm::linearRand(0.0f, width));
    posY.push_back(glm::linearRand(0.0f, height));
    posZ.push_back(glm::linearRand(0.0f, depth));

    velX.push_back(randDir.x * speed);
    velY.push_back(randDir.y * speed);
    velZ.push_back(randDir.z * speed);

    if (isShark) {
        type.push_back(1);
    }
    else {
        type.push_back(0);
    }
}


void SimulationCPU::update(float dt)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    globalTime += dt;

    auto t_sort_start = std::chrono::high_resolution_clock::now();
    // Grid reset
    std::fill(cellStart.begin(), cellStart.end(), -1);
    std::fill(cellEnd.begin(), cellEnd.end(), -1);

    // Assign to grid
    for (int i = 0; i < numBoids; ++i) {
        gridHash[i] = getCellID(posX[i], posY[i], posZ[i]);
        gridIndex[i] = i;
    }

    // Sort grid
    std::vector<int> p(numBoids);
    for (int i = 0; i < numBoids; i++) p[i] = i;

    std::sort(p.begin(), p.end(), [&](int a, int b) {
        return gridHash[a] < gridHash[b];
        });

    std::vector<int> sortedHash(numBoids);
    std::vector<int> sortedIndex(numBoids);
    for (int i = 0; i < numBoids; ++i) {
        sortedHash[i] = gridHash[p[i]];
        sortedIndex[i] = gridIndex[p[i]];
    }

    // Find start and end of cells
    for (int i = 0; i < numBoids; ++i)
    {
        int hash = sortedHash[i];

        if (i == 0 || hash != sortedHash[i - 1]) {
            cellStart[hash] = i;
        }
        if (i == numBoids - 1 || hash != sortedHash[i + 1]) {
            cellEnd[hash] = i + 1;
        }
    }
    auto t_sort_end = std::chrono::high_resolution_clock::now();

    auto t_sim_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numBoids; ++i)
    {
        float pos_x = posX[i];
        float pos_y = posY[i];
        float pos_z = posZ[i];

        float vel_x = velX[i];
        float vel_y = velY[i];
        float vel_z = velZ[i];

        int my_type = type[i];
        float currentMaxSpeed = maxSpeed;

        if (my_type == 1) // Shark
        {
            currentMaxSpeed = sharkMaxSpeed;

            // Random wander direction
            glm::vec3 wanderTarget = getWanderTarget_cpu(i, globalTime, 0.5f);

            float steerForce = 2 * sharkMaxSpeed;

            vel_x += wanderTarget.x * steerForce * dt;
            vel_y += wanderTarget.y * steerForce * dt;
            vel_z += wanderTarget.z * steerForce * dt;
        }
        else // fish
        {
            // Empty accumulative data
            float xpos_avg = 0.0f, ypos_avg = 0.0f, zpos_avg = 0.0f;
            float xvel_avg = 0.0f, yvel_avg = 0.0f, zvel_avg = 0.0f;
            float close_dx = 0.0f, close_dy = 0.0f, close_dz = 0.0f;
            float neighbors_count = 0.0f;

            bool predator_detected = false;
            float pred_dx_sum = 0.0f, pred_dy_sum = 0.0f, pred_dz_sum = 0.0f;

            int grid_x = (int)(pos_x / cellSize);
            int grid_y = (int)(pos_y / cellSize);
            int grid_z = (int)(pos_z / cellSize);

            // Search neighboring cells
            for (int z = -1; z <= 1; z++) {
                for (int y = -1; y <= 1; y++) {
                    for (int x = -1; x <= 1; x++) {
                        int neighbor_x = grid_x + x;
                        int neighbor_y = grid_y + y;
                        int neighbor_z = grid_z + z;

                        if (neighbor_x < 0 || neighbor_x >= gridWidth ||
                            neighbor_y < 0 || neighbor_y >= gridHeight ||
                            neighbor_z < 0 || neighbor_z >= gridDepth) continue;

                        int cell_id = (neighbor_z * gridHeight * gridWidth) + (neighbor_y * gridWidth) + neighbor_x;
                        int start = cellStart[cell_id];
                        if (start == -1) continue;
                        int end = cellEnd[cell_id];

                        for (int k = start; k < end; ++k)
                        {
                            int otherIdx = sortedIndex[k];
                            if (i == otherIdx) continue;

                            float other_px = posX[otherIdx];
                            float other_py = posY[otherIdx];
                            float other_pz = posZ[otherIdx];

                            float dx = pos_x - other_px;
                            float dy = pos_y - other_py;
                            float dz = pos_z - other_pz;

                            float distSq = dx * dx + dy * dy + dz * dz;

                            // Shark visible range
                            if (distSq < visualRangeSq * predatorDetectionMultiplier * predatorDetectionMultiplier)
                            {
                                int neighborType = type[otherIdx];

                                if (neighborType == 1) // Shark detected
                                {
                                    // Distance to shark
                                    pred_dx_sum += dx;
                                    pred_dy_sum += dy;
                                    pred_dz_sum += dz;
                                    predator_detected = true;
                                }
                                else // Flocking
                                {
                                    if (distSq < protectedRangeSq) {
                                        // Separation
                                        close_dx += dx;
                                        close_dy += dy;
                                        close_dz += dz;
                                    }
                                    else if (distSq < visualRangeSq) {
                                        // Aligment
                                        xpos_avg += other_px;
                                        ypos_avg += other_py;
                                        zpos_avg += other_pz;

                                        // Cohesion
                                        xvel_avg += velX[otherIdx];
                                        yvel_avg += velY[otherIdx];
                                        zvel_avg += velZ[otherIdx];

                                        neighbors_count += 1.0f;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (neighbors_count > 0.0f) {
                xpos_avg /= neighbors_count;
                ypos_avg /= neighbors_count;
                zpos_avg /= neighbors_count;

                xvel_avg /= neighbors_count;
                yvel_avg /= neighbors_count;
                zvel_avg /= neighbors_count;

                // Cohesion: Move toward center
                vel_x += (xpos_avg - pos_x) * centeringFactor;
                vel_y += (ypos_avg - pos_y) * centeringFactor;
                vel_z += (zpos_avg - pos_z) * centeringFactor;

                // Alignment: Match velocity
                vel_x += (xvel_avg - vel_x) * matchingFactor;
                vel_y += (yvel_avg - vel_y) * matchingFactor;
                vel_z += (zvel_avg - vel_z) * matchingFactor;
            }

            // Separation: Avoid other boids
            vel_x += close_dx * avoidFactor;
            vel_y += close_dy * avoidFactor;
            vel_z += close_dz * avoidFactor;

            // Shark avoidance
            if (predator_detected) {
                float pTurn = avoidFactor * 2.0f;

                if (pred_dx_sum > 0) vel_x += pTurn;
                if (pred_dx_sum < 0) vel_x -= pTurn;

                if (pred_dy_sum > 0) vel_y += pTurn;
                if (pred_dy_sum < 0) vel_y -= pTurn;

                if (pred_dz_sum > 0) vel_z += pTurn;
                if (pred_dz_sum < 0) vel_z -= pTurn;
            }
        }

        applyTurnFactor(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z);
        applyRepulsion(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z);
        applySpeedLimits(vel_x, vel_y, vel_z, my_type == 0 ? maxSpeed : sharkMaxSpeed);

        // Update position
        n_posX[i] = pos_x + vel_x * dt;
        n_posY[i] = pos_y + vel_y * dt;
        n_posZ[i] = pos_z + vel_z * dt;

        n_velX[i] = vel_x;
        n_velY[i] = vel_y;
        n_velZ[i] = vel_z;
    }

    posX = n_posX;
    posY = n_posY;
    posZ = n_posZ;
    velX = n_velX;
    velY = n_velY;
    velZ = n_velZ;

    auto t_sim_end = std::chrono::high_resolution_clock::now();

    m_stats.timeSort = std::chrono::duration<float, std::milli>(t_sort_end - t_sort_start).count();
    m_stats.timeUpdate = std::chrono::duration<float, std::milli>(t_sim_end - t_sim_start).count();

    m_stats.timeCopy = 0.0f;

    auto t_end = std::chrono::high_resolution_clock::now();
    m_stats.timeTotal = std::chrono::duration<float, std::milli>(t_end - t_start).count();
}

int SimulationCPU::getCellID(float x, float y, float z) const
{
    int cx = (int)(x / cellSize);
    int cy = (int)(y / cellSize);
    int cz = (int)(z / cellSize);

    if (cx < 0) cx = 0; if (cx >= gridWidth)  cx = gridWidth - 1;
    if (cy < 0) cy = 0; if (cy >= gridHeight) cy = gridHeight - 1;
    if (cz < 0) cz = 0; if (cz >= gridDepth)  cz = gridDepth - 1;

    return (cz * gridHeight * gridWidth) + (cy * gridWidth) + cx;
}

void SimulationCPU::applyTurnFactor(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z)
{
    if (pos_x < margin) vel_x += turnFactor;
    if (pos_x > width - margin) vel_x -= turnFactor;
    if (pos_y < margin) vel_y += turnFactor;
    if (pos_y > height - margin) vel_y -= turnFactor;
    if (pos_z < margin) vel_z += turnFactor;
    if (pos_z > depth - margin) vel_z -= turnFactor;
}

void SimulationCPU::applySpeedLimits(float& vel_x, float& vel_y, float& vel_z, float maxSpeed)
{
    float speedSq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;

    if (speedSq > maxSpeed * maxSpeed) {
        float s = std::sqrt(speedSq);
        vel_x = (vel_x / s) * maxSpeed;
        vel_y = (vel_y / s) * maxSpeed;
        vel_z = (vel_z / s) * maxSpeed;
    }
    else if (speedSq < minSpeed * minSpeed) {
        float s = std::sqrt(speedSq);
        if (s > 0.00001f) {
            vel_x = (vel_x / s) * minSpeed;
            vel_y = (vel_y / s) * minSpeed;
            vel_z = (vel_z / s) * minSpeed;
        }
    }
}

// Set mouse interaction data
void SimulationCPU::setInteractionData(bool active, float origin_x, float origin_y, float origin_z, float dir_x, float dir_y, float dir_z)
{
    this->isRepulsorActive = active;
    this->origin = glm::vec3(origin_x, origin_y, origin_z);
    this->dir = glm::vec3(dir_x, dir_y, dir_z);
}

// Apply mouse interaction
void SimulationCPU::applyRepulsion(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z)
{
    if (!isRepulsorActive) return;

    const float beamRadius = 150.0f;

    float toBoidX = pos_x - origin.x;
    float toBoidY = pos_y - origin.y;
    float toBoidZ = pos_z - origin.z;
    
    float t = toBoidX * dir.x + toBoidY * dir.y + toBoidZ * dir.z;

    if (t > 0.0f)
    {
        float closeX = origin.x + t * dir.x;
        float closeY = origin.y + t * dir.y;
        float closeZ = origin.z + t * dir.z;

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