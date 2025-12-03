#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <shader_m.h> 

#include "SimulationCPU.h"
#include "SimulationGPU.h"

#include "imgui\imgui.h"
#include "imgui\imgui_impl_glfw.h"
#include "imgui\imgui_impl_opengl3.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <chrono>

enum class SimulationMode {
    CPU,
    GPU
};

// CONFIG PARAMETERS
SimulationMode CURRENT_MODE = SimulationMode::GPU; // CHANGE FOR GPU / CPU
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

int BOIDS = 10000;

const float WORLD_HEIGHT = 2000.0f;
const float WORLD_WIDTH = 2000.0f;
const float WORLD_DEPTH = 2000.0f;

// CAMERA PARAMETERS
glm::vec3 cameraPos = glm::vec3(SCR_WIDTH / 2, SCR_HEIGHT / 2, 1200.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

bool firstMouse = true;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
float yaw = -90.0f;
float pitch = 0.0f;

float fov = 45.0f;

glm::vec3 interactionOrigin = glm::vec3(0.0f);
glm::vec3 interactionDir = glm::vec3(0.0f);
bool isInteractionActive = false;

struct RenderContext {
    unsigned int VAO;

    unsigned int VBO_PosX;
    unsigned int VBO_PosY;
    unsigned int VBO_PosZ;

    unsigned int VBO_VelX;
    unsigned int VBO_VelY;
    unsigned int VBO_VelZ;

    unsigned int VBO_type;
};

unsigned int boxVAO = 0, boxVBO = 0;
Shader* boxShader = nullptr;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

void setupRendering3D(RenderContext& ctx);
void setupBox();

void processInput(GLFWwindow* window, float dt);
void renderUI(const std::unique_ptr<ISimulation>& simPtr);

void updateSimulation(const std::unique_ptr<ISimulation>& simPtr, float dt); 
void updateInteraction(GLFWwindow* window, const glm::mat4& view, const glm::mat4& proj, std::unique_ptr<ISimulation>& simPtr);

void uploadCPUDataToGPU(const std::unique_ptr<ISimulation>& simPtr, const RenderContext& ctx);

void drawBox(const glm::mat4& view, const glm::mat4& proj, Shader& shader);

void clearScreen();

void renderScene(const RenderContext& ctx, const Shader& shader, const glm::mat4& view, const glm::mat4& proj);
int main(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-mode" && i + 1 < argc) {
            std::string modeStr = argv[i + 1];
            if (modeStr == "CPU") {
                CURRENT_MODE = SimulationMode::CPU;
            }
            else if (modeStr == "GPU") {
                CURRENT_MODE = SimulationMode::GPU;
            }
            else {
                std::cout << "Warning: Unknown mode '" << modeStr << "'. Using default (GPU)." << std::endl;
            }
            i++;
        }
        else if (arg == "-n" && i + 1 < argc) {
            try {
                BOIDS = std::stoi(argv[i + 1]);
                if (BOIDS < 0) BOIDS = 100;
            }
            catch (...) {
                std::cout << "Warning: Invalid number for -n. Using default (10000)." << std::endl;
            }
            i++;
        }
    }

    std::cout << "--- SIMULATION CONFIG ---" << std::endl;
    std::cout << "Mode: " << (CURRENT_MODE == SimulationMode::GPU ? "GPU" : "CPU") << std::endl;
    std::cout << "Boids: " << BOIDS << std::endl;
    std::cout << "-------------------------" << std::endl;

    // GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Boids Simulation 3D", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSwapInterval(0); // V-Sync Off

    // GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    glEnable(GL_DEPTH_TEST);

    Shader boidShader("shaders/boid.vs", "shaders/boid.fs");
    boxShader = new Shader("shaders/box.vs", "shaders/box.fs");

    std::unique_ptr<ISimulation> simPtr = nullptr;

    if (CURRENT_MODE == SimulationMode::GPU) {
        cudaSetDevice(0);
    }

    RenderContext ctx;
    setupRendering3D(ctx);
    setupBox();

    // Get simulation pointer
    switch (CURRENT_MODE)
    {
        case SimulationMode::CPU:
            simPtr = std::make_unique<SimulationCPU>(BOIDS, WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH);
            break;
        case SimulationMode::GPU:
            simPtr = std::make_unique<SimulationGPU>(BOIDS, WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH);
            dynamic_cast<SimulationGPU*>(simPtr.get())->registerVBOs(
                ctx.VBO_PosX, ctx.VBO_PosY, ctx.VBO_PosZ,
                ctx.VBO_VelX, ctx.VBO_VelY, ctx.VBO_VelZ, ctx.VBO_type
            );
            break;
    }

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = (float)glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window, deltaTime);

        clearScreen();
        renderUI(simPtr);

        glm::mat4 proj = glm::perspective(glm::radians(fov), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 5000.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        updateInteraction(window, view, proj, simPtr);

        updateSimulation(simPtr, deltaTime);

        if (CURRENT_MODE == SimulationMode::CPU)
        {
            uploadCPUDataToGPU(simPtr, ctx);
        }

        drawBox(view, proj, *boxShader);

        renderScene(ctx, boidShader, view, proj);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &ctx.VAO);
    glDeleteBuffers(1, &ctx.VBO_PosX); 
    glDeleteBuffers(1, &ctx.VBO_PosY); 
    glDeleteBuffers(1, &ctx.VBO_PosZ);
    glDeleteBuffers(1, &ctx.VBO_VelX); 
    glDeleteBuffers(1, &ctx.VBO_VelY); 
    glDeleteBuffers(1, &ctx.VBO_VelZ);
    glDeleteBuffers(1, &ctx.VBO_type);

    glfwTerminate();
    return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
        firstMouse = true;
        return;
    }

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    cameraFront = glm::normalize(front);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    fov -= (float)yoffset;
    if (fov < 1.0f) fov = 1.0f;
    if (fov > 90.0f) fov = 90.0f;
}

void processInput(GLFWwindow* window, float dt) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
    else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    float speed = 1000.0f * dt;

    // Camera movement
    // Z-
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += speed * cameraFront;
    // Z+
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= speed * cameraFront;
    // X-
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
    // X+
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * speed;
    // Y+
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += speed * glm::vec3(0.0f, 1.0f, 0.0f);
    // Y-
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        cameraPos -= speed * glm::vec3(0.0f, 1.0f, 0.0f);
}

void setupRendering3D(RenderContext& ctx)
{
    // Boid Geometry
    float vertices[] = {
         0.0f,  10.0f,  0.0f,   -0.6f, 0.5f, 0.6f,
        -4.0f, -5.0f,  3.0f,    -0.6f, 0.5f, 0.6f,
         0.0f, -5.0f, -2.0f,    -0.6f, 0.5f, 0.6f,

          0.0f,  10.0f,  0.0f,    0.6f, 0.5f, 0.6f,
          0.0f, -5.0f, -2.0f,     0.6f, 0.5f, 0.6f,
          4.0f, -5.0f,  3.0f,     0.6f, 0.5f, 0.6f,

           0.0f,  10.0f,  0.0f,    0.0f, 0.2f, 0.9f,
           4.0f, -5.0f,  3.0f,     0.0f, -0.8f, 0.6f,
          -4.0f, -5.0f,  3.0f,     0.0f, -0.8f, 0.6f,

          -4.0f, -5.0f,  3.0f,     0.0f, -0.5f, -0.8f,
           4.0f, -5.0f,  3.0f,     0.0f, -0.5f, -0.8f,
           0.0f, -5.0f, -2.0f,     0.0f, -0.5f, -0.8f
    };

    glGenVertexArrays(1, &ctx.VAO);
    unsigned int VBO_mesh;
    glGenBuffers(1, &VBO_mesh);

    glGenBuffers(1, &ctx.VBO_PosX); 
    glGenBuffers(1, &ctx.VBO_PosY); 
    glGenBuffers(1, &ctx.VBO_PosZ);
    glGenBuffers(1, &ctx.VBO_VelX); 
    glGenBuffers(1, &ctx.VBO_VelY); 
    glGenBuffers(1, &ctx.VBO_VelZ);
    glGenBuffers(1, &ctx.VBO_type);

    glBindVertexArray(ctx.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_mesh);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(7);

    auto setupAttrib = [](unsigned int vbo, int loc) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, BOIDS * sizeof(float), NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
        glVertexAttribDivisor(loc, 1);
        };

    setupAttrib(ctx.VBO_PosX, 1);
    setupAttrib(ctx.VBO_PosY, 2);
    setupAttrib(ctx.VBO_PosZ, 3);

    setupAttrib(ctx.VBO_VelX, 4);
    setupAttrib(ctx.VBO_VelY, 5);
    setupAttrib(ctx.VBO_VelZ, 6);

    glBindBuffer(GL_ARRAY_BUFFER, ctx.VBO_type);
    glBufferData(GL_ARRAY_BUFFER, BOIDS * sizeof(int), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(8); 
    glVertexAttribIPointer(8, 1, GL_INT, 0, (void*)0);
    glVertexAttribDivisor(8, 1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void renderUI(const std::unique_ptr<ISimulation>& simPtr) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Simulation Settings");
    ImGui::Text("Mode: %s", CURRENT_MODE == SimulationMode::GPU ? "GPU" : "CPU");
    ImGui::Text("Number od boids: %d", BOIDS);

    SimulationStats stats = simPtr->getStats();

    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Frame Time: %.3f ms", 1000.0f / ImGui::GetIO().Framerate);
    ImGui::Separator();

    ImGui::Text("Performance Breakdown:");
    ImGui::Text("Initialization (One-time): %.3f ms", stats.timeInit);
    ImGui::Text("Data Copy/Map:             %.3f ms", stats.timeCopy);
    ImGui::Text("Grid Sort/Hash:            %.3f ms", stats.timeSort);
    ImGui::Text("Physics Calculation:       %.3f ms", stats.timeUpdate);
    ImGui::Separator();
    ImGui::Text("Total Simulation Step:     %.3f ms", stats.timeTotal);
    ImGui::Separator();

    bool changed = false;
    switch (CURRENT_MODE) {
        case SimulationMode::GPU:
        {
            auto simGPU = dynamic_cast<SimulationGPU*>(simPtr.get());
            if (!simGPU) break;
            auto& p = simGPU->getParamsRef();
            changed |= ImGui::SliderFloat("Visual Range", &p.visualRange, 20.0f, 100.0f);
            changed |= ImGui::SliderFloat("Shark Detection Range Multiplier", &p.predatorDetectionMultiplier, 0.0f, 4.0f);
            changed |= ImGui::SliderFloat("Min Speed", &p.minSpeed, 1.0f, 50.0f);
            changed |= ImGui::SliderFloat("Max Speed", &p.maxSpeed, 1.0f, 50.0f);
            changed |= ImGui::SliderFloat("Turn Factor", &p.turnFactor, 0.0f, 10.0f);
            changed |= ImGui::SliderFloat("Cohesion", &p.centeringFactor, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Alignment", &p.matchingFactor, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Separation", &p.avoidFactor, 0.0f, 1.0f);
            if (changed)
                simGPU->updateParamsOnGPU();
            break;
        }
        case SimulationMode::CPU:
        {
            auto simCPU = dynamic_cast<SimulationCPU*>(simPtr.get());
            ImGui::SliderFloat("Min Speed", &simCPU->minSpeed, 1.0f, 50.0f);
            ImGui::SliderFloat("Max Speed", &simCPU->maxSpeed, 1.0f, 50.0f);
            ImGui::SliderFloat("Turn Factor", &simCPU->turnFactor, 0.1f, 10.0f);
            ImGui::SliderFloat("Cohesion", &simCPU->centeringFactor, 0.0f, 0.01f, "%.5f");
            ImGui::SliderFloat("Alignment", &simCPU->matchingFactor, 0.0f, 1.0f);
            ImGui::SliderFloat("Separation", &simCPU->avoidFactor, 0.0f, 1.0f);
            ImGui::SliderFloat("Repulsion", &simCPU->repulsionStrength, 0.0f, 500.0f);

            break;
        }
            
    }
    ImGui::End();
}

void updateSimulation(const std::unique_ptr<ISimulation>& simPtr, float dt) {
    simPtr->update(dt);
}

void uploadCPUDataToGPU(const std::unique_ptr<ISimulation>& simPtr, const RenderContext& ctx) {

    auto cpuSim = dynamic_cast<SimulationCPU*>(simPtr.get());
    if (!cpuSim) return;

    auto upload = [&](unsigned int vbo, const std::vector<float>& data) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(float), data.data());
        };

    upload(ctx.VBO_PosX, cpuSim->getPosX());
    upload(ctx.VBO_PosY, cpuSim->getPosY());
    upload(ctx.VBO_PosZ, cpuSim->getPosZ());
    upload(ctx.VBO_VelX, cpuSim->getVelX());
    upload(ctx.VBO_VelY, cpuSim->getVelY());
    upload(ctx.VBO_VelZ, cpuSim->getVelZ());

    glBindBuffer(GL_ARRAY_BUFFER, ctx.VBO_type);
    const std::vector<int>& types = cpuSim->getType();
    glBufferSubData(GL_ARRAY_BUFFER, 0, types.size() * sizeof(int), types.data());
}

void renderScene(const RenderContext& ctx, const Shader& shader, const glm::mat4& view, const glm::mat4& proj) {
    shader.use();
    shader.setMat4("projection", proj);
    shader.setMat4("view", view);

    glBindVertexArray(ctx.VAO);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 12, BOIDS);
    glBindVertexArray(0);
}
void clearScreen()
{
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void setupBox() {
    float vertices[] = {
        0.0f, 0.0f, 0.0f,  // 0
        WORLD_WIDTH, 0.0f, 0.0f,  // 1
        WORLD_WIDTH, WORLD_HEIGHT, 0.0f,  // 2
        0.0f, WORLD_HEIGHT, 0.0f,  // 3
        0.0f, 0.0f, WORLD_DEPTH,  // 4
        WORLD_WIDTH, 0.0f, WORLD_DEPTH,  // 5
        WORLD_WIDTH, WORLD_HEIGHT, WORLD_DEPTH,  // 6
        0.0f, WORLD_HEIGHT, WORLD_DEPTH   // 7
    };

    unsigned int indices[] = {
        0, 1, 1, 2, 2, 3, 3, 0, // Front face
        4, 5, 5, 6, 6, 7, 7, 4, // Back face
        0, 4, 1, 5, 2, 6, 3, 7  // Connecting lines
    };

    unsigned int EBO;
    glGenVertexArrays(1, &boxVAO);
    glGenBuffers(1, &boxVBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(boxVAO);

    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void drawBox(const glm::mat4& view, const glm::mat4& proj, Shader& shader) {
    shader.use();

    shader.setMat4("view", view);
    shader.setMat4("projection", proj);

    glm::mat4 model = glm::mat4(1.0f);

    glBindVertexArray(boxVAO);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void updateInteraction(GLFWwindow* window, const glm::mat4& view, const glm::mat4& proj, std::unique_ptr<ISimulation>& simPtr)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        isInteractionActive = true;
    }
    else {
        isInteractionActive = false;
        return;
    }

    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);

    float x = (2.0f * (float)mouseX) / SCR_WIDTH - 1.0f;
    float y = 1.0f - (2.0f * (float)mouseY) / SCR_HEIGHT;

    glm::vec4 ray_clip = glm::vec4(x, y, -1.0, 1.0);
    glm::vec4 ray_eye = glm::inverse(proj) * ray_clip;
    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);
    glm::vec3 ray_wor = glm::vec3(glm::inverse(view) * ray_eye);

    interactionDir = glm::normalize(ray_wor);
    interactionOrigin = cameraPos;

    if (simPtr) {
        float o_x = interactionOrigin.x;
        float o_y = interactionOrigin.y;
        float o_z = interactionOrigin.z;
        float d_x = interactionDir.x;
        float d_y = interactionDir.y;
        float d_z = interactionDir.z;

        simPtr->setInteractionData(isInteractionActive, o_x, o_y, o_z, d_x, d_y, d_z);
    }
}
