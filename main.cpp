#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <chrono>
#include <thread>

#include "glad/glad.h"
#include <glfw3.h>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <imgui.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Shader.hpp"
#include "Camera.hpp"
#include "VertexBuffer.hpp"
#include "VertexArray.hpp"
#include "Textures.hpp"
#include "ComputeShader.hpp"
#include "IndexBuffer.hpp"
#include "ShaderBuffer.hpp"
#include "Solver.hpp"

#include <filesystem>

#ifdef _WIN32
    #include <windows.h>
#elif __linux__
    #include <unistd.h>
#elif __APPLE__
    #include <mach-o/dyld.h>
#endif



void framebuffer_size_callback(GLFWwindow* window, int width, int heigth);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void changeBorderWidth(VertexArray& borderVao, Shader& shader);
void changeBorderHeight(VertexArray& borderVao, Shader& shader);
void updateBuffers(ShaderBuffer<glm::vec2>& particlesLocationBuffer);
void populateArray(std::vector<int>& arr, int size);
std::string getExecutablePath();
std::string getProjectRootPath();

void imguiMenu(float deltaTime, VertexArray& borderVao, Shader& render, Shader& borderS, ComputeShader& compute);


static float vertices_wTexture[] = {

    -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
     1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 0.0f, 1.0f, 0.0f,

};

static float border[] = {
  
    -1.f, 1.f, 0.f,
    -1.f, -1.f, 0.f,
     1.f, -1.f, 0.f,
     1.f, 1.f, 0.f,
    -1.f, 1.f, 0.f

};


static float WIDTH = 1600;
static float HEIGHT = 900;

// Particles Radius
static float smoothingRadius = 100.f;
static float particleRadius = 10.f;
static float sampleDensity = 0.0f;

// Border Width and Height
static float W_H[] = {WIDTH, HEIGHT};

// Mouse Pos 
static glm::vec2 mousePos;

// Resize Variable
static bool resize = false;

// Border boundries but not normalized between -1 - 1 !!!! index 0 = width_start, index 1 = width_end, index 2 = height_start, index 3 = height_end 
static float borderPixelCoords[] = {(WIDTH/2- W_H[0]/2), (WIDTH/2 + W_H[0]/2), (HEIGHT/2 - W_H[1]/2), (HEIGHT/2 + W_H[1]/2)};

// Solver
static std::unique_ptr<Solver> solver;

// Density Radius
static glm::vec2 densityCenterPnt;
static bool isPressed = false;

// Draw grid
static bool drawGrid = false;

// Change Sim
static bool changeSim = false;

int main(){

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL", NULL, NULL);

    if (window == NULL){

        std::cout << "Failed to create a window" << "\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_UNAVAILABLE);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Uncapped FPS
    glfwSwapInterval(60);

    // Setting cursor function callback
    glfwSetCursorPosCallback(window, mouse_callback);

    // Setting scroll wheel funcion callback
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){

        std::cout << "Failed to initialize GLAD" << "\n";
        return EXIT_FAILURE;
    }


    // Setting viewport
    glViewport(0, 0, WIDTH, HEIGHT);

    // ImGui Setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Init imgui for OpenGL
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");


    // Retriving Relative Paths
    std::string projectRoot = getProjectRootPath();
    std::cout << "Project Root Path: " << projectRoot << std::endl;

    std::string shaderDir = projectRoot + "/shaders";
    std::cout << "Shader Directory: " << shaderDir << std::endl;

    // Shaders
    Shader renderS                  ((shaderDir + "/vertex.vert").c_str()       , (shaderDir + "/fragment.frag").c_str());
    Shader borderS                  ((shaderDir + "/vertexBorder.vert").c_str() , (shaderDir + "/fragmentBorder.frag").c_str());

    // Compute Shaders
    ComputeShader solver2D          ((shaderDir + "/2DSolver.comp").c_str());
    ComputeShader solver2D_test     ((shaderDir + "/2DSolver_test.comp").c_str());
    ComputeShader GPUhashing        ((shaderDir + "/SpatialHash.comp").c_str());
    ComputeShader GPUsort           ((shaderDir + "/GPUsort.comp").c_str());

    

    VertexArray VAO;
    VertexBuffer VBO(vertices_wTexture, sizeof(vertices_wTexture));
    VAO.addLayout(GL_FLOAT, 3, false);
    VAO.addBuffer(VBO, 5 * sizeof(float));
    VAO.Unbind();

    VertexArray borderVAO;
    VertexBuffer borderVBO(border, sizeof(border));
    borderVAO.addLayout(GL_FLOAT, 3, false);
    borderVAO.addBuffer(borderVBO);
    borderVAO.Unbind();

    // Initilizing Solver
    solver = std::make_unique<Solver>(borderPixelCoords);

    // Shader uniforms
    renderS.useShader();
    renderS.setFloat("smoothingRadius", solver->getSmoothingRadius());
    renderS.setVec4fv("border", {borderPixelCoords[0], borderPixelCoords[1], borderPixelCoords[2], borderPixelCoords[3]});
    renderS.setBool("drawGrid", drawGrid);
    borderS.useShader();
    borderS.setFloat("smoothingRadius", solver->getSmoothingRadius());
    
    // Timing 
    float deltaTime = 0.0f; // time between current frame and last frame
    float lastFrame = 0.0f; // time of last frame
    int fCounter = 0;

    //timing variable
    float neghboorSearch = 0;
    float writingToBuffer = 0;
    float dispatchCompute = 0;
    float count = 0;

    // GPU sort Testing
    // ShaderBuffer<int> toSort(12);
    // std::vector<int> array;

    // int size = 64;
    // populateArray(array, size);
    // toSort.setData(array.data(), array.size() * sizeof(int));


    // GPUsort.useShader();
    // GPUsort.setUint("plCount", size);
   
    // // int numStages = (int)std::log2(131072); // next pow of 2 over 100k
    // int numStages = 6;

    // for (int stageIndex = 0; stageIndex < numStages; stageIndex++)
    // {
    //     for (int stepIndex = 0; stepIndex < stageIndex + 1; stepIndex++)
    //     {
    //         // GPUsort.useShader();
    //         // Calculate some pattern stuff
    //         int groupWidth = 1 << (stageIndex - stepIndex);
    //         int groupHeight = 2 * groupWidth - 1;
    //         GPUsort.setInt("groupWidth", groupWidth);
    //         GPUsort.setInt("groupHeight", groupHeight);
    //         GPUsort.setInt("stepIndex", stepIndex);
    //         // Run the sorting step on the GPU
    //         glDispatchCompute(size, 1, 1);
    //         glMemoryBarrier(GL_ALL_BARRIER_BITS);
    //     }
    // }

    // toSort.getData(array, size);

    // for(int i = 0; i < size - 1; i++)
    //     if(!(array[i] <= array[i+1])){
    //         std::cout << "List sort was unsuccesfully" << std::endl;
    //         break;
    //     }
    // std::cout << "List sorted succesfully" << std::endl;


    // Render Loop
    while(!glfwWindowShouldClose(window)){

        if (solver->getParticlesLocationCount())
            count++;
        else
        {
            count = 0;
            neghboorSearch = 0;
            writingToBuffer = 0;
            dispatchCompute = 0;
        }

        // Input 
        processInput(window);

        // Set frame time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        if (resize){
            changeBorderWidth(borderVAO, renderS);
            changeBorderHeight(borderVAO, renderS);
            // normalizeParticlesPos(particlesLocationBuffer);
            resize = false;
        }



        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // glEnable(GL_DEPTH_TEST);


        ImGui_ImplOpenGL3_NewFrame();        
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        // solver->Solve(deltaTime);
        solver->ComputeShaderSolve(deltaTime, solver2D, neghboorSearch, writingToBuffer, dispatchCompute);
        // solver->ComputeShaderSolveGpuSort(deltaTime, solver2D_test, GPUsort, GPUhashing, neghboorSearch, writingToBuffer, dispatchCompute);
            


        // ---------- Render particles ----------
        renderS.useShader();
        renderS.setVec2fv("gridNum", solver->getGridNum());
        renderS.setInt("particlesIndex", 1);
        renderS.setFloat("smoothingRadius", solver->getSmoothingRadius());
        renderS.setIntArray("neighbooringIDs", solver->getNeighbooringOffesets(), 9);
        VAO.Bind();
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        VAO.Unbind();
        // --------------------------------------

    
        // ---------- Render Border---------- 
        borderS.useShader();
        borderVAO.Bind();
        glLineWidth(1);
        glDrawArrays(GL_LINE_STRIP, 0, 5);
        // --------------------------------------
        
        
        // WireFrame mode
        // glPolygonMode(GL_FRONT, GL_LINE);
        // glPolygonMode(GL_BACK, GL_LINE);

        imguiMenu(deltaTime, borderVAO, renderS, borderS, solver2D);
        // imguiTestMenu(deltaTime, borderVAO, particlesLocationBuffer, , renderS, exampleS, computeS);
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        
        
        // check and call events and swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    
    return EXIT_SUCCESS;
}

void imguiMenu(float deltaTime, VertexArray& borderVao, Shader& render, Shader& shaderB, ComputeShader& compute){

    // Number of particles to add 
    static int v = 1000;
    static float p_to_Add[2] = {0, 0};
    static float spacing = 20;
    static int particlesTadd = 400;
    static float pressure = solver->getPressureMultiplier();
    static float nearPressure = solver->getPressureNearMultiplier();
    static float density = solver->getTargetDensity();
    static float sigma = solver->getSigma();
    static float beta = solver->getBeta();
    static float attractionForce = solver->getAttractionForce();

    
    static bool gravityOn = true;
    static float g = abs(solver->getGravity().y);

    ImVec2 defaultWindowSize(200.0f, 400.0f);
    ImGui::SetWindowSize(defaultWindowSize, ImGuiCond_FirstUseEver);

    //ImGui
    ImGui::Begin("Fluid Simulation");
    ImGui::Text("FPS: %1.f", 1/deltaTime);
    ImGui::Text("Density: %.5f", sampleDensity);
    ImGui::Text("mouseX: %1.f, mouseY: %1.f", mousePos.x, mousePos.y);
    
    ImGui::PushItemWidth(120.0f);

    if (ImGui::SliderFloat("Border Width", W_H, 0, WIDTH, "%.1f")){
        changeBorderWidth(borderVao, render);
    }

    if (ImGui::SliderFloat("Border Height", (W_H + 1), 0, HEIGHT, "%.1f")){
        changeBorderHeight(borderVao, render);
    }

    if (ImGui::SliderFloat("Smoothing Radius", &smoothingRadius, 1, WIDTH/2)){

        render.useShader();
        render.setFloat("smoothingRadius", smoothingRadius);
        shaderB.useShader();
        shaderB.setFloat("smoothingRadius", smoothingRadius);
        compute.useShader();
        compute.setFloat("smoothingRadius", smoothingRadius);

        solver->setSmoothingRadius(smoothingRadius);
    }

    if (ImGui::SliderFloat("Pressure constant", &pressure, 0.f, 300.f)){
        solver->setPressureMultiplier(pressure);
    }

    if (ImGui::SliderFloat("Near Pressure constant", &nearPressure, 0.f, 300.f)){
        solver->setPressureNearMultiplier(nearPressure);
    }

    if (ImGui::SliderFloat("Target Density", &density, 0.f, 80.f)){
        solver->setTargetDensity(density);
    }

    if (ImGui::SliderFloat("Gravity", &g, 0.f, 20.f)){
        solver->setGravity({0.0f, -g});
    }

    if (ImGui::SliderFloat("Sigma", &sigma, 0.f, 4.f)){
        solver->setSigma(sigma);
    }

    if (ImGui::SliderFloat("Beta", &beta, 0.00001f, 0.05f)){
        solver->setBeta(beta);
    }

    if (ImGui::SliderFloat("Attraction Force", &attractionForce, 0.f, 50.f)){
        solver->setAttraction(attractionForce);
    }


    ImGui::Text("--------------------");

    ImGui::SliderInt("Particles to spawn", &v, 0, 5000);

    if (ImGui::Button("Random Location Spawn"))
    {
        std::vector<glm::vec2> pl;
        pl.resize(v);

        std::random_device rd("1");
        std::uniform_int_distribution<int> distW(borderPixelCoords[0] + particleRadius, borderPixelCoords[1] - particleRadius);
        std::uniform_int_distribution<int> distH(borderPixelCoords[2] + particleRadius, borderPixelCoords[3] - particleRadius);

        for(int i = 0; i < v; i++){
            pl[i].x = distW(rd);
            pl[i].y = distH(rd);
        }

        
        auto start = std::chrono::high_resolution_clock::now();
        solver->setParticlesLocation(pl);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Set particles time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }   

    ImGui::Text("--------------------");

    if (ImGui::Button("Toggle Gravity")){

        gravityOn = !gravityOn;
        
        if (gravityOn){
            solver->setGravity({0.0, -g});
        }
        else 
            solver->setGravity({0.0, 0.0});

    }


    if (ImGui::Button("Toggle Grid")){

        drawGrid = !drawGrid;
        render.useShader();
        render.setBool("drawGrid", drawGrid);
    }

    ImGui::Text("--------------------");

    if (ImGui::Button("Clear Particles"))
    {
        solver->clearParticles();
    }

    if (ImGui::Button("Change Sim")){
        changeSim = !changeSim;
    }

    ImGui::End();
}


void updateBuffers(ShaderBuffer<glm::vec2>& particlesLocationBuffer){

    if(solver->getParticlesLocations().empty()) return;

    particlesLocationBuffer.setData(solver->getParticlesLocations().data(), solver->getParticlesLocationsSize());
    // .setData(solver->getDensityData(), solver->getDensitySize());


}

void framebuffer_size_callback(GLFWwindow* window, int width, int heigth){
    glViewport(0, 0, width, heigth);

    WIDTH = width;
    HEIGHT = heigth;
    // Uncomment/comment if you want after the resize to stay to the same border dimensions
    // W_H[0] = WIDTH;
    // W_H[1] = HEIGHT;
    resize = true;

}

void processInput(GLFWwindow* window) {

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS){
        solver->toggleAttraction(true, mousePos);
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_RELEASE){
        solver->toggleAttraction(false, mousePos);
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS)
        isPressed = false;

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
    {
        // sampleDensity = solver->calculateDensity(mousePos);
        // std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

void mouse_callback(GLFWwindow* window, double xPos, double yPos){

    mousePos = {xPos, HEIGHT - yPos};

}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){

}

void changeBorderWidth(VertexArray& borderVao, Shader& shader){
    float rel_Width = (W_H[0]/WIDTH);
        
        border[0] = -rel_Width;
        border[3] = -rel_Width;
        border[6] = rel_Width;
        border[9] = rel_Width;
        border[12] = -rel_Width;

        borderVao.setData(border, sizeof(border));

        borderPixelCoords[0] = (WIDTH/2- W_H[0]/2);
        borderPixelCoords[1] = (WIDTH/2 + W_H[0]/2);
        
        // Calculating border Coords
        shader.useShader();
        shader.setVec4fv("border", {borderPixelCoords[0], borderPixelCoords[1], borderPixelCoords[2], borderPixelCoords[3]});

        // Sending change to solver
        solver->setBorder(borderPixelCoords);
}

void changeBorderHeight(VertexArray& borderVao, Shader& shader){
    float rel_height = (W_H[1]/HEIGHT);
        
        border[1] = rel_height;
        border[4] = -rel_height;
        border[7] = -rel_height;
        border[10] = rel_height;
        border[13] = rel_height;

        borderVao.setData(border, sizeof(border));

        borderPixelCoords[2] = (HEIGHT/2 - W_H[1]/2);
        borderPixelCoords[3] = (HEIGHT/2 + W_H[1]/2);

        // Calculating border Coords
        shader.useShader();
        shader.setVec4fv("border", {borderPixelCoords[0], borderPixelCoords[1], borderPixelCoords[2], borderPixelCoords[3]});
        
        // Sending change to solver
        solver->setBorder(borderPixelCoords);
}


void populateArray(std::vector<int>& arr, int size){

    arr.resize(size);
    std::random_device rd("1");
    std::uniform_int_distribution<int> dist(0, 100);

    for(int i = 0; i < size; i++){
        arr[i] = dist(rd);
    }
}

std::string getExecutablePath() {
    char buffer[1024];
    std::string execPath;

#ifdef _WIN32
    // For Windows
    GetModuleFileNameA(NULL, buffer, sizeof(buffer));
    execPath = std::string(buffer);
#elif __linux__
    // For Linux
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        execPath = std::string(buffer);
    }
#elif __APPLE__
    // For macOS
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        execPath = std::string(buffer);
    }
#endif

    return execPath;
}

std::string getProjectRootPath() {
    std::string execPath = getExecutablePath();

    // Use filesystem to get the directory path and navigate to the project folder
    std::filesystem::path execDir = std::filesystem::path(execPath).parent_path();

    // Assuming the executable is inside 'build/' or a similar directory
    // and the project root is one level above
    std::filesystem::path projectRoot = execDir.parent_path().parent_path();  // Adjust accordingly

    return projectRoot.string();
}