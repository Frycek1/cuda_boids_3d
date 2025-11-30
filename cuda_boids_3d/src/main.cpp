#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <shader_m.h> 

#include "SimulationCPU.h"
#include "SimulationGPU.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>

// --- KONFIGURACJA ---
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

const int BOIDS = 50000;

const float WORLD_HEIGHT = 2000.0f;
const float WORLD_WIDTH = 2000.0f;
const float WORLD_DEPTH = 2000.0f;

enum class SimulationMode {
    CPU,
    GPU
};

const SimulationMode CURRENT_MODE = SimulationMode::GPU;

// --- KAMERA 3D ---
glm::vec3 cameraPos = glm::vec3(SCR_WIDTH / 2, SCR_HEIGHT / 2, 1200.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

struct RenderContext {
    unsigned int VAO;

    unsigned int VBO_PosX;
    unsigned int VBO_PosY;
    unsigned int VBO_PosZ;

    unsigned int VBO_VelX;
    unsigned int VBO_VelY;
    unsigned int VBO_VelZ;
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window, float dt);
void setupRendering3D(RenderContext& ctx);

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Boids 3D Simulation", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(0);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    Shader boidShader("shaders/boid.vs", "shaders/boid.fs");

    // --- INICJALIZACJA SYMULACJI ---

    bool isGPUMode = (CURRENT_MODE == SimulationMode::GPU);

    // --- SETUP GPU (BUFORY) ---
    RenderContext ctx;
    setupRendering3D(ctx);

    std::unique_ptr<ISimulation> simPtr = nullptr;
    cudaSetDevice(0);

    switch (CURRENT_MODE)
    {
        case SimulationMode::CPU:
            simPtr = std::make_unique<SimulationCPU>(BOIDS, WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH);
            break;
        case SimulationMode::GPU:
            simPtr = std::make_unique<SimulationGPU>(BOIDS, WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH);
            dynamic_cast<SimulationGPU*>(simPtr.get())->registerVBOs(
                ctx.VBO_PosX, ctx.VBO_PosY, ctx.VBO_PosZ,
                ctx.VBO_VelX, ctx.VBO_VelY, ctx.VBO_VelZ
            );

            break;
    }

    // --- PĘTLA GŁÓWNA ---
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    double fpsLastTime = glfwGetTime();
    int fpsNbFrames = 0;

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = (float)glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // FPS w tytule
        fpsNbFrames++;
        if (currentFrame - fpsLastTime >= 1.0) {
            std::string modeStr = isGPUMode ? "[GPU CUDA]" : "[CPU GRID]";
            std::string title = modeStr + " FPS: " + std::to_string(fpsNbFrames) + " | Boids: " + std::to_string(BOIDS);
            glfwSetWindowTitle(window, title.c_str());
            fpsNbFrames = 0;
            fpsLastTime = currentFrame;
        }

        // Input (Kamera)
        processInput(window, deltaTime);

        // Renderowanie
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Czyścimy też bufor głębokości

        boidShader.use();

        // Macierze 3D
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 5000.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        boidShader.setMat4("projection", projection);
        boidShader.setMat4("view", view);

        glBindVertexArray(ctx.VAO);

        // --- AKTUALIZACJA FIZYKI ---

        if (SimulationMode::GPU == CURRENT_MODE)
        {
            simPtr->update(deltaTime);

        }
        else
        {
            simPtr->update(deltaTime);
            auto cpuSim = dynamic_cast<SimulationCPU*>(simPtr.get());
            auto uploadVBO = [&](unsigned int vbo, const std::vector<float>& data) {
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, BOIDS * sizeof(float), data.data());
                };

            uploadVBO(ctx.VBO_PosX, cpuSim->getPosX());
            uploadVBO(ctx.VBO_PosY, cpuSim->getPosY());
            uploadVBO(ctx.VBO_PosZ, cpuSim->getPosZ()); // <-- Z

            uploadVBO(ctx.VBO_VelX, cpuSim->getVelX());
            uploadVBO(ctx.VBO_VelY, cpuSim->getVelY());
            uploadVBO(ctx.VBO_VelZ, cpuSim->getVelZ()); // <-- VZ

        }
        glDrawArraysInstanced(GL_TRIANGLES, 0, 12, BOIDS);

        glBindVertexArray(0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Sprzątanie
    // Destruktory simGPU/simCPU wywołają się automatycznie
    glDeleteVertexArrays(1, &ctx.VAO);
    glDeleteBuffers(1, &ctx.VBO_PosX); glDeleteBuffers(1, &ctx.VBO_PosY); glDeleteBuffers(1, &ctx.VBO_PosZ);
    glDeleteBuffers(1, &ctx.VBO_VelX); glDeleteBuffers(1, &ctx.VBO_VelY); glDeleteBuffers(1, &ctx.VBO_VelZ);

    glfwTerminate();
    return 0;
}

// --- IMPLEMENTACJE ---

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window, float dt) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float speed = 1000.0f * dt; // Prędkość kamery

    // Poruszanie w płaszczyźnie XZ
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= speed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;

    // Winda (Góra/Dół bezwzględna)
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cameraPos += speed * glm::vec3(0.0f, 1.0f, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPos -= speed * glm::vec3(0.0f, 1.0f, 0.0f);
}

void setupRendering3D(RenderContext& ctx)
{
    // Model: Ostrosłup (Tetrahedron)
    // Dziób wzdłuż osi Y (bo tak boidy latają domyślnie)
    float vertices[] = {
        // Ściana 1 (Lewa)
        0.0f,  10.0f,  0.0f,   // Dziób
       -4.0f, -5.0f,  3.0f,   // Lewy tył
        0.0f, -5.0f, -2.0f,   // Grzbiet tył

        // Ściana 2 (Prawa)
        0.0f,  10.0f,  0.0f,   // Dziób
        0.0f, -5.0f, -2.0f,   // Grzbiet tył
        4.0f, -5.0f,  3.0f,   // Prawy tył

        // Ściana 3 (Dół)
        0.0f,  10.0f,  0.0f,
        4.0f, -5.0f,  3.0f,
       -4.0f, -5.0f,  3.0f,

       // Ściana 4 (Tył - "Silnik")
      -4.0f, -5.0f,  3.0f,
       4.0f, -5.0f,  3.0f,
       0.0f, -5.0f, -2.0f
    };

    glGenVertexArrays(1, &ctx.VAO);
    unsigned int VBO_mesh;
    glGenBuffers(1, &VBO_mesh);

    // Generujemy 6 buforów instancji
    glGenBuffers(1, &ctx.VBO_PosX); 
    glGenBuffers(1, &ctx.VBO_PosY); 
    glGenBuffers(1, &ctx.VBO_PosZ);
    glGenBuffers(1, &ctx.VBO_VelX); 
    glGenBuffers(1, &ctx.VBO_VelY); 
    glGenBuffers(1, &ctx.VBO_VelZ);

    glBindVertexArray(ctx.VAO);

    // 0. Mesh Geometry (nie zmienia się)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_mesh);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Helper do ustawiania atrybutów
    auto setupAttrib = [](unsigned int vbo, int loc) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, BOIDS * sizeof(float), NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
        glVertexAttribDivisor(loc, 1); // Per instance
        };

    setupAttrib(ctx.VBO_PosX, 1);
    setupAttrib(ctx.VBO_PosY, 2);
    setupAttrib(ctx.VBO_PosZ, 3);

    setupAttrib(ctx.VBO_VelX, 4);
    setupAttrib(ctx.VBO_VelY, 5);
    setupAttrib(ctx.VBO_VelZ, 6);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}