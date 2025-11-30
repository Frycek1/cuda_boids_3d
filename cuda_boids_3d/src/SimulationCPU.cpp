#include "SimulationCPU.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace glm;

SimulationCPU::SimulationCPU(int numBoids, float width, float height, float depth) : screenWidth(width), screenHeight(height), screenDepth(depth)
{
    this->numBoids = numBoids;

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

    positions_x.reserve(numBoids); 
    positions_y.reserve(numBoids); 
    positions_z.reserve(numBoids);
    velocities_x.reserve(numBoids); 
    velocities_y.reserve(numBoids); 
    velocities_z.reserve(numBoids);

    for (int i = 0; i < numBoids; i++)
    {
        createNewBoid();
    }

    cellSize = visualRange;

    gridWidth = (int)std::ceil(screenWidth / cellSize);
    gridHeight = (int)std::ceil(screenHeight / cellSize);
    gridDepth = (int)std::ceil(screenDepth / cellSize);

    numCells = gridWidth * gridHeight * gridDepth;

    // Resize
    boid_cellIDs.resize(numBoids);
    boid_indices.resize(numBoids);
    cell_start_indices.resize(numCells);

    n_posX.resize(numBoids); n_posY.resize(numBoids); n_posZ.resize(numBoids);
    n_velX.resize(numBoids); n_velY.resize(numBoids); n_velZ.resize(numBoids);
}

void SimulationCPU::createNewBoid()
{
    vec3 randDir = glm::sphericalRand(1.0f);
    float speed = (minspeed + maxspeed) * 0.5f;

    positions_x.push_back(glm::linearRand(0.0f, screenWidth));
    positions_y.push_back(glm::linearRand(0.0f, screenHeight));
    positions_z.push_back(glm::linearRand(0.0f, screenDepth)); // <-- Z

    velocities_x.push_back(randDir.x * speed);
    velocities_y.push_back(randDir.y * speed);
    velocities_z.push_back(randDir.z * speed); // <-- Z
}

void SimulationCPU::applyScreenEdges(int i)
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

void SimulationCPU::enforceSpeedLimits(int i)
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

void SimulationCPU::applyBias(int i)
{
    // (Opcjonalne - bez zmian, tylko dodaj obsługę Z jeśli chcesz bias w 3D)
}

int SimulationCPU::getCellID(float x, float y, float z) const
{
    int cx = (int)(x / cellSize);
    int cy = (int)(y / cellSize);
    int cz = (int)(z / cellSize);

    // Clamp
    if (cx < 0) cx = 0; if (cx >= gridWidth)  cx = gridWidth - 1;
    if (cy < 0) cy = 0; if (cy >= gridHeight) cy = gridHeight - 1;
    if (cz < 0) cz = 0; if (cz >= gridDepth)  cz = gridDepth - 1;

    // 1D Index z 3D
    return (cz * gridHeight * gridWidth) + (cy * gridWidth) + cx;
}

void SimulationCPU::update(float dt)
{
    // 1. Reset Grid
    std::fill(cell_start_indices.begin(), cell_start_indices.end(), -1);

    // 2. Assign Cell ID
    for (int i = 0; i < numBoids; ++i) {
        boid_cellIDs[i] = getCellID(positions_x[i], positions_y[i], positions_z[i]);
        boid_indices[i] = i;
    }

    // 3. Sort (CPU Sort - tu może być wąskie gardło przy dużej ilości boidów)
    std::vector<int> p(numBoids);
    for (int i = 0; i < numBoids; i++) p[i] = i;

    std::sort(p.begin(), p.end(), [&](int a, int b) {
        return boid_cellIDs[a] < boid_cellIDs[b];
        });

    // Reorder 'temp' arrays for keys and indices
    std::vector<int> sortedIDs(numBoids), sortedIndices(numBoids);
    for (int i = 0; i < numBoids; ++i) {
        sortedIDs[i] = boid_cellIDs[p[i]];
        sortedIndices[i] = boid_indices[p[i]];
    }
    boid_cellIDs = sortedIDs;
    boid_indices = sortedIndices;

    // 4. Find Starts
    for (int i = 0; i < numBoids; ++i) {
        int cell = boid_cellIDs[i];
        if (i == 0 || cell != boid_cellIDs[i - 1]) {
            cell_start_indices[cell] = i;
        }
    }

    // 5. Physics Update
    // Uwaga: w przeciwieństwie do GPU, tutaj nie robimy pełnego Reorderu danych fizycznych,
    // bo na CPU cache działa inaczej (i jest trudniej to zrobić ręcznie bez spadku wydajności przez kopiowanie).
    // Skaczemy po pamięci (boid_indices), co na CPU jest znośne przy małej liczbie boidów (<10k).

    for (int i = 0; i < numBoids; ++i)
    {
        float px = positions_x[i];
        float py = positions_y[i];
        float pz = positions_z[i];
        float vx = velocities_x[i];
        float vy = velocities_y[i];
        float vz = velocities_z[i];

        float avgVX = 0, avgVY = 0, avgVZ = 0;
        float avgPX = 0, avgPY = 0, avgPZ = 0;
        float closeDX = 0, closeDY = 0, closeDZ = 0;
        float count = 0;

        int myCX = (int)(px / cellSize);
        int myCY = (int)(py / cellSize);
        int myCZ = (int)(pz / cellSize);

        // Pętla 3x3x3 po sąsiadach
        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {

                    int nX = myCX + x;
                    int nY = myCY + y;
                    int nZ = myCZ + z;

                    if (nX < 0 || nX >= gridWidth || nY < 0 || nY >= gridHeight || nZ < 0 || nZ >= gridDepth)
                        continue;

                    int nKey = (nZ * gridHeight * gridWidth) + (nY * gridWidth) + nX;
                    int start = cell_start_indices[nKey];

                    if (start == -1) continue;

                    // Iteracja po boidach w komórce
                    for (int k = start; k < numBoids; ++k)
                    {
                        if (boid_cellIDs[k] != nKey) break; // Koniec komórki

                        int otherIdx = boid_indices[k];
                        if (i == otherIdx) continue;

                        float opx = positions_x[otherIdx];
                        float opy = positions_y[otherIdx];
                        float opz = positions_z[otherIdx];

                        float dx = px - opx;
                        float dy = py - opy;
                        float dz = pz - opz;

                        float distSq = dx * dx + dy * dy + dz * dz;

                        if (distSq < visualRangeSq) {
                            if (distSq < protectedRangeSq) {
                                closeDX += dx; closeDY += dy; closeDZ += dz;
                            }
                            else {
                                avgVX += velocities_x[otherIdx];
                                avgVY += velocities_y[otherIdx];
                                avgVZ += velocities_z[otherIdx];
                                avgPX += opx; avgPY += opy; avgPZ += opz;
                                count += 1.0f;
                            }
                        }
                    }
                }
            }
        }

        if (count > 0) {
            avgVX /= count; avgVY /= count; avgVZ /= count;
            avgPX /= count; avgPY /= count; avgPZ /= count;

            vx += (avgVX - vx) * matchingfactor;
            vy += (avgVY - vy) * matchingfactor;
            vz += (avgVZ - vz) * matchingfactor;

            vx += (avgPX - px) * centeringfactor;
            vy += (avgPY - py) * centeringfactor;
            vz += (avgPZ - pz) * centeringfactor;
        }

        vx += closeDX * avoidfactor;
        vy += closeDY * avoidfactor;
        vz += closeDZ * avoidfactor;

        // Zapis tymczasowy
        velocities_x[i] = vx; velocities_y[i] = vy; velocities_z[i] = vz;

        applyScreenEdges(i);
        enforceSpeedLimits(i);

        n_posX[i] = px + velocities_x[i] * dt;
        n_posY[i] = py + velocities_y[i] * dt;
        n_posZ[i] = pz + velocities_z[i] * dt;
        n_velX[i] = velocities_x[i];
        n_velY[i] = velocities_y[i];
        n_velZ[i] = velocities_z[i];
    }

    positions_x = n_posX; positions_y = n_posY; positions_z = n_posZ;
    velocities_x = n_velX; velocities_y = n_velY; velocities_z = n_velZ;
}