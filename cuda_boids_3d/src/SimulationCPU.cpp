#include "SimulationCPU.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace glm;

SimulationCPU::SimulationCPU(int numBoids, float width, float height, float depth) : width(width), height(height), depth(depth)
{
    this->numBoids = numBoids;

    this->depth = height;

    visualRange = 40.0f;
    protectedRange = 8.0f;
    centeringFactor = 0.0005f;
    avoidFactor = 0.05f;
    matchingFactor = 0.05f;
    turnFactor = 0.5f;
    margin = 100.0f;
    maxSpeed = 6.0f;
    minSpeed = 3.0f;
    visualRangeSq = visualRange * visualRange;
    protectedRangeSq = protectedRange * protectedRange;

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
}

void SimulationCPU::createNewBoid()
{
    vec3 randDir = glm::sphericalRand(1.0f);
    float Speed = (minSpeed + maxSpeed) * 0.5f;

    posX.push_back(glm::linearRand(0.0f, width));
    posY.push_back(glm::linearRand(0.0f, height));
    posZ.push_back(glm::linearRand(0.0f, depth));

    velX.push_back(randDir.x * Speed);
    velY.push_back(randDir.y * Speed);
    velZ.push_back(randDir.z * Speed);
}


void SimulationCPU::update(float dt)
{
    std::fill(cellStart.begin(), cellStart.end(), -1);
    std::fill(cellEnd.begin(), cellEnd.end(), -1);

    // 2. Assign Cell ID
    for (int i = 0; i < numBoids; ++i) {
        gridHash[i] = getCellID(posX[i], posY[i], posZ[i]);
        gridIndex[i] = i;
    }

    // 3. Sort (CPU Sort - tu może być wąskie gardło przy dużej ilości boidów)
    std::vector<int> p(numBoids);
    for (int i = 0; i < numBoids; i++) p[i] = i;

    std::sort(p.begin(), p.end(), [&](int a, int b) {
        return gridHash[a] < gridHash[b];
        });

    // Reorder 'temp' arrays for keys and indices
    std::vector<int> sortedIDs(numBoids), sortedIndices(numBoids);
    for (int i = 0; i < numBoids; ++i) {
        sortedIDs[i] = gridHash[p[i]];
        sortedIndices[i] = gridIndex[p[i]];
    }
    gridHash = sortedIDs;
    gridIndex = sortedIndices;

    // 4. Find Starts
    for (int i = 0; i < numBoids; ++i)
    {
        int hash = gridHash[i];

        if (i == 0 || hash != gridHash[i - 1]) {
            cellStart[hash] = i;
        }
        if (i == numBoids - 1 || hash != gridHash[i + 1]) {
            cellEnd[hash] = i + 1;
        }
    }


    for (int i = 0; i < numBoids; ++i)
    {
        float pos_x = posX[i];
        float pos_y = posY[i];
        float pos_z = posZ[i];

        float vel_x = velX[i];
        float vel_y = velY[i];
        float vel_z = velZ[i];

        float close_dx = 0.0f, close_dy = 0.0f, close_dz = 0.0f;
        float pos_x_avg = 0.0f, pos_y_avg = 0.0f, pos_z_avg = 0.0f;
        float vel_x_avg = 0.0f, vel_y_avg = 0.0f, vel_z_avg = 0.0f;
        float neighbors_count = 0.0f;

        int grid_x = (int)(pos_x / cellSize);
        int grid_y = (int)(pos_y / cellSize);
        int grid_z = (int)(pos_z / cellSize);

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
                    int start_id = cellStart[cell_id];

                    if (start_id == -1) continue;
                    int end_id = cellEnd[cell_id];

                    for (int k = start_id; k < end_id; ++k)
                    {

                        int otherIdx = gridIndex[k];
                        if (i == otherIdx) continue;

                        float other_px = posX[otherIdx];
                        float other_py = posY[otherIdx];
                        float other_pz = posZ[otherIdx];

                        float dx = pos_x - other_px;
                        float dy = pos_y - other_py;
                        float dz = pos_z - other_pz;

                        float distSq = dx * dx + dy * dy + dz * dz;

                        if (distSq < visualRangeSq) {
                            if (distSq < protectedRangeSq) {
                                close_dx += dx; 
                                close_dy += dy; 
                                close_dz += dz;
                            }
                            else {
                                vel_x_avg += velX[otherIdx];
                                vel_y_avg += velY[otherIdx];
                                vel_z_avg += velZ[otherIdx];

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

        if (neighbors_count > 0) {
            vel_x_avg /= neighbors_count; 
            vel_y_avg /= neighbors_count; 
            vel_z_avg /= neighbors_count;

            pos_x_avg /= neighbors_count; 
            pos_y_avg /= neighbors_count; 
            pos_z_avg /= neighbors_count;

            vel_x += (vel_x_avg - vel_x) * matchingFactor;
            vel_y += (vel_y_avg - vel_y) * matchingFactor;
            vel_z += (vel_z_avg - vel_z) * matchingFactor;

            vel_x += (pos_x_avg - pos_x) * centeringFactor;
            vel_y += (pos_y_avg - pos_y) * centeringFactor;
            vel_z += (pos_z_avg - pos_z) * centeringFactor;
        }

        vel_x += close_dx * avoidFactor;
        vel_y += close_dy * avoidFactor;
        vel_z += close_dz * avoidFactor;

        applyTurnFactor(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z);
        applyRepulsion(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z);
        applySpeedLimits(vel_x, vel_y, vel_z);

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

void SimulationCPU::applySpeedLimits(float& vel_x, float& vel_y, float& vel_z)
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

void SimulationCPU::setInteractionData(bool active, float origin_x, float origin_y, float origin_z, float dir_x, float dir_y, float dir_z)
{
    this->isRepulsorActive = active;
    this->origin = glm::vec3(origin_x, origin_y, origin_z);
    this->dir = glm::vec3(dir_x, dir_y, dir_z);
}

void SimulationCPU::applyRepulsion(float pos_x, float pos_y, float pos_z, float& vel_x, float& vel_y, float& vel_z)
{
    if (!isRepulsorActive) return;

    const float beamRadius = 150.0f;
    const float repulsionStrength = 200.0f;

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