#pragma once 

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono> 
#include <windows.h>

#include "ShaderBuffer.hpp"

#include <glm/glm.hpp>

class Solver {

    // Particle Locations
    std::vector<glm::vec2> particlesLocations;  
    std::vector<glm::vec2> particlesPredictedLocations;  

    // Size in bytes of the vector holding all particle locations
    unsigned int plSize;
    unsigned int plCount;

    // Spatial Lookup
    std::vector<glm::ivec2> spatialLookUp;
    std::vector<int> spatialOffsets;

    // Buffers for compute shader
    ShaderBuffer<glm::vec2> particlesLocationBuffer;
    ShaderBuffer<glm::uint32> bufferIndexesByCellID;
    ShaderBuffer<glm::u32vec2> bufferStartEndIndexes;
    ShaderBuffer<glm::vec2> bufferVelocities;
    ShaderBuffer<glm::vec2> bufferPrevVelocities;
    ShaderBuffer<glm::vec2> bufferPrevPositions;
    ShaderBuffer<glm::vec2> bufferDensities;
    ShaderBuffer<glm::uvec2> debugBufferCounter;
    ShaderBuffer<glm::vec4> debugPressure;
    ShaderBuffer<int> debugID;
    ShaderBuffer<glm::ivec2> spatialLookupBuffer;
    ShaderBuffer<int> spatialOffsetsBuffer;
    
    // Holding previous iteration locations to calculate the velocity
    std::vector<glm::vec2> particlesPrevLocations;
    std::vector<glm::vec2> velocities;
    std::vector<int> nearBorderParticlesIndex;
    glm::vec2 initialVelocity;

    // Gravity
    glm::vec2 gravity;
    // Springs
    std::vector<glm::vec2> springs;  

    // Each particle local density 
    std::vector<float> particlesDensities;
    // Target Density
    float targetDensity;
    float pressureMultiplier;
    float pressureNearMultiplier;

    // Near Densities
    std::vector<float> particlesNearDensities;
    
    // Dictionary that holds particles in cells by IDs
    mutable std::unordered_map<int, std::vector<int>> pointGridDict;
    // Dictionary that holds the location of neighboor particles for a cell by IDs
    mutable std::unordered_map<int, std::vector<int>> IDGridDict;
    mutable std::vector<uint32_t> indexesByCellID;
    mutable std::vector<glm::u32vec2> startEndIndexes;


    // Border Coordinates
    glm::vec2 m_borderW; 
    glm::vec2 m_borderH;

    // Holding the total number of cells for X and Y
    glm::ivec2 gridNum;

    // Particle "interference" radius 
    float m_smoothingRadius;
    float diameter;
    // Holding offsets to find neighbooring cells with ID 
    int neighbooringIDs[9];

    // Threads
    std::vector<std::thread> threads;
    int totalThreads;

    // Mutex for synchronizing access to getNeighboors
    mutable std::mutex neighborsMutex;
    mutable std::mutex densityMutex;

    //Particles Radius
    float particlesRadius = 10.f;

    // Viscocity constants
    float sigma;
    float beta;

    // Attraction Force
    float attraction;
    bool isPressed;
    glm::vec2 mousePos;

public:
    //Debuging Timers
    float spatialHashGetNeigh = 0;
    float dictionaryHashNeigh = 0;
    int neighboorCounter = 0;


    
    Solver(){}

    // No particles constuctor
    Solver(float border[4], float smoothingRadius = 90.f, glm::vec2 initVel = {0, 0}, float pressureMult = 150, float pressureNearMulti = 70, float atrctForce = 15.f)
    : m_smoothingRadius {smoothingRadius},
      particlesLocationBuffer(0),
      bufferIndexesByCellID(1),
      bufferStartEndIndexes(2),
      bufferVelocities(3),
      bufferPrevPositions(4),
      bufferDensities(5),
      debugBufferCounter(6),
      debugID(7),
      bufferPrevVelocities(8),
      debugPressure(9),
      spatialLookupBuffer(10),
      spatialOffsetsBuffer(11)

    {
        particlesLocations = {};
        particlesPredictedLocations = {};
        particlesPrevLocations = {};
        velocities = {};
        nearBorderParticlesIndex = {};
        springs = {};
        spatialLookUp = {};
        spatialOffsets = {};
        plSize = 0;
        plCount = 0;

        initialVelocity = initVel;        
        pressureMultiplier = pressureMult;
        pressureNearMultiplier = pressureNearMulti;
        attraction = atrctForce;
        targetDensity = 35;
        gravity = {0, -25.f};
        sigma = 0.02f;
        beta = 0.005f;
        isPressed = false;
        mousePos = {0, 0};

        setBorder(border);
        calculateGrid();
        initilizePrevPos();

        totalThreads = std::thread::hardware_concurrency();
    }


    void Solve(float timeStep){
        parrallel(timeStep);
        cacheParticles();
    }

    void ComputeShaderSolve(float dt, ComputeShader& compute, float& neighboorsTiming, float& writeBuffersTiming, float& computeTiming){

        // dt = 0.03;
        if(plCount == 0) return;
        
        // Retrieve new locations from GPU to calculate neighbors on CPU
        particlesLocationBuffer.getData(particlesLocations, plCount);

        // Calculate Neighbors for compute on CPU
        computeShadersCalcuateNeighboors();

        auto end = std::chrono::high_resolution_clock::now();
        // Set buffers with new neighbors
        // glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        
        bufferIndexesByCellID.setData(indexesByCellID.data(), indexesByCellID.size() * sizeof(glm::uint32));
        bufferStartEndIndexes.setData(startEndIndexes.data(), startEndIndexes.size() * sizeof(glm::uvec2));

        // Ensure all CPU writes are visible to GPU
        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            
        // Set Uniforms
        compute.useShader();
        compute.setUint("plCount", plCount);
        compute.setVec4fv("border", {m_borderW, m_borderH});
        compute.setVec2iv("gridNum", gridNum);
        compute.setFloat("dt", dt);
        compute.setFloat("targetDensity", targetDensity);
        compute.setFloat("pressMult", pressureMultiplier);
        compute.setFloat("nearPressMult", pressureNearMultiplier);
        compute.setFloat("smoothingRadius", m_smoothingRadius);
        compute.setFloat("gravity", gravity.y);
        compute.setFloat("sigma", sigma);
        compute.setFloat("beta", beta);
        compute.setFloat("attraction", attraction);
        compute.setBool("isClicked", isPressed);
        compute.setVec2fv("mousePos", mousePos);

        
        // {

        //     GLuint query;
        //     glGenQueries(1, &query);

        //     // Begin the query
        //     glBeginQuery(GL_TIME_ELAPSED, query);


        //     // call glDispatchCompute here
        //     glDispatchCompute(plCount/64 + plCount%64, 1, 1);


        //     // End the query
        //     glEndQuery(GL_TIME_ELAPSED);

        //     // Wait until the results are available (optional)
        //     GLint available = 0;
        //     while (!available) {
        //         glGetQueryObjectiv(query, GL_QUERY_RESULT_AVAILABLE, &available);
        //     }

        //     // Get the query result
        //     GLuint64 elapsed_time = 0;
        //     glGetQueryObjectui64v(query, GL_QUERY_RESULT, &elapsed_time);

        //     double elapsed_time_ms = static_cast<double>(elapsed_time) / 1e6;

        //     // Print the elapsed time in milliseconds
        //     printf("Elapsed time: %.2f ms\n", elapsed_time_ms);

        //     // Delete the query object when done
        //     glDeleteQueries(1, &query);
        // }
        // Call compute Shader
        glDispatchCompute(plCount/64 + plCount%64, 1, 1);
        // compareIDs();
        // compareDensities();
        // compareDensityRelaxations();
        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // glDispatchCompute(plCount/2, plCount/2, 1);
        // glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // verifyNeighbooringArrays();
        // Retriving new location to calculate neighboors

    
        // End of compute
    }

    void ComputeShaderSolveGpuSort(float dt, ComputeShader& s_compute, ComputeShader& s_gpuSort, ComputeShader& s_spatialHash, float& neighboorsTiming, float& writeBuffersTiming, float& computeTiming){

        // dt = 0.01;
        if(plCount == 0) return;


        // Retriving particles Locations for testing
        // particlesLocationBuffer.getData(particlesLocations, plCount);

        // Populating Hash Table
        s_spatialHash.useShader();
        s_spatialHash.setUint("plCount", plCount);
        s_spatialHash.setVec4fv("border", {m_borderW, m_borderH});
        s_spatialHash.setVec2iv("gridNum", gridNum);
        s_spatialHash.setFloat("smoothingRadius", m_smoothingRadius);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glDispatchCompute(plCount/64 + plCount%64, 1, 1);

        // Sorting Hash Table and Writing offsets
        // gpuSort(s_gpuSort);

        // ----------- //

        // initSpatialHash();

        std::vector<glm::ivec2> gpuSpatialHashArray;
        gpuSpatialHashArray.resize(plCount);
        spatialLookupBuffer.getData(gpuSpatialHashArray, plCount);

        std::sort(gpuSpatialHashArray.begin(), gpuSpatialHashArray.end(), &Solver::compareSpatialLookUp);

        spatialLookupBuffer.setData(gpuSpatialHashArray.data(), plCount * sizeof(glm::ivec2));
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        s_gpuSort.useShader();
        s_gpuSort.setUint("plCount", plCount);
        s_gpuSort.setBool("setOffsets", true);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glDispatchCompute(plCount/64 + plCount%64, 1, 1);


        // std::vector<int> gpuSpatialHashOffArray;
        // gpuSpatialHashOffArray.resize(plCount);
        // spatialOffsetsBuffer.getData(gpuSpatialHashOffArray, plCount);

        // for(int i = 0; i < plCount; i++){
            // if(gpuSpatialHashOffArray[i] != spatialOffsets[i])
            // {
            //     std::cout << "Offset arrays mismatch, index: " << i << std::endl;
            // } 
            // if(gpuSpatialHashArray[i].x != spatialLookUp[i].x)
            // {
            //     std::cout << "Lookup arrays mismatch, index: " << i << std::endl;
            // } 
        // }
        
        // for(int i = 0; i < plCount; i++){
        //     if(gpuSpatialHashArray[i].x != spatialLookUp[i].x)
        //     {
        //         std::cout << "Lookup arrays mismatch, index: " << i << std::endl;
        //     } 
        // }
        // std::cout << "#################"<< std::endl;


        // ----------- //

        // Set Uniforms
        s_compute.useShader();
        s_compute.setUint("plCount", plCount);
        s_compute.setVec4fv("border", {m_borderW, m_borderH});
        s_compute.setVec2iv("gridNum", gridNum);
        s_compute.setIntArray("neighbooringIDs", neighbooringIDs, 9);
        s_compute.setFloat("dt", dt);
        s_compute.setFloat("targetDensity", targetDensity);
        s_compute.setFloat("pressMult", pressureMultiplier);
        s_compute.setFloat("nearPressMult", pressureNearMultiplier);
        s_compute.setFloat("smoothingRadius", m_smoothingRadius);
        s_compute.setFloat("gravity", gravity.y);
        s_compute.setFloat("sigma", sigma);
        s_compute.setFloat("beta", beta);
        s_compute.setFloat("attraction", attraction);
        s_compute.setBool("isClicked", isPressed);
        s_compute.setVec2fv("mousePos", mousePos);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glDispatchCompute(plCount/64 + plCount%64, 1, 1);

    
    }

    void gpuSort(ComputeShader& gpuSort){

        gpuSort.useShader();
        gpuSort.setUint("plCount", plCount);
        gpuSort.setBool("setOffsets", false);
   
        int numStages = (int)std::log2(nextPowOf_2(plCount));

        for (int stageIndex = 0; stageIndex < numStages; stageIndex++)
        {
            for (int stepIndex = 0; stepIndex < stageIndex + 1; stepIndex++)
            {

                // Calculate some pattern stuff
                int groupWidth = 1 << (stageIndex - stepIndex);
                int groupHeight = 2 * groupWidth - 1;
                gpuSort.setInt("groupWidth", groupWidth);
                gpuSort.setInt("groupHeight", groupHeight);
                gpuSort.setInt("stepIndex", stepIndex);
                // Run the sorting step on the GPU
                glDispatchCompute((plCount/64 + plCount%64)/2, 1, 1);
                glMemoryBarrier(GL_ALL_BARRIER_BITS);
            }

            // TODO::add uniform to calculate offsets 
            if (stageIndex == numStages - 1){
                gpuSort.setBool("setOffsets", true);
                glDispatchCompute(plCount/64 + plCount%64, 1, 1);
            }

        }
    }

    int nextPowOf_2(int value){
        
        for(int i = 0; i < 30; i++)
            if (pow(2, i) > value)
                return pow(2, i);

    }

    void compareDensityRelaxations(){
        int count = 0;
        std::vector<glm::vec4> GPUpressure(plCount, glm::vec4(-1));
        std::vector<glm::uvec2> counter (1, glm::uvec2(0));
        debugBufferCounter.getData(counter, 1);

        debugPressure.getData(GPUpressure, plCount);
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);

        for(int i = 0; i < plCount; i++){

           if(GPUpressure[i].x - GPUpressure[i].z > 0.001 || GPUpressure[i].y - GPUpressure[i].w > 0.001){
                std::cout << "Index: " << i << "  ALL: (" << GPUpressure[i].z << ", " << GPUpressure[i].w << ") "
                    << ", NEIGH: (" << GPUpressure[i].x << ", " << GPUpressure[i].y << "), " << std::endl;
                    count++;
           }

        }

        std::cout << "\n----------------------" << std::endl;
        std::cout << "ALL : " << counter[0].y << ", NEIGH: " << counter[0].x << std::endl;
        std::cout << "Correct: " << plCount - count << ", False: " << count << std::endl;
        std::cout << "----------------------" << std::endl;

    }

    void compareIDs(){

        std::vector<int> gpuIDs(plCount, -1);
        int count = 0;
        
        debugID.getData(gpuIDs, plCount);
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        
        for(int i = 0; i < plCount; i++){

            if(getID(particlesLocations[i]) != gpuIDs[i]){
                
                std::cout << "CPU ID: " << getID(particlesLocations[i])
                    << ", GPU ID: " << gpuIDs[i] << std::endl; 
                    count++;
            }
            // assert(gpuIDs[i] >= 0);

        }

        std::cout << "\n----------------------" << std::endl;
        std::cout << "Correct: " << plCount - count << ", False: " << count << std::endl;
        std::cout << "----------------------" << std::endl;
    }

    void calculateDensitiesCPU(std::vector<glm::vec2>& densities){

        glm::ivec2 counter {0, 0};
        for(int j = 0; j < plCount; j++){

            glm::vec2 currentP = particlesLocations[j];
            int currID = getID(currentP);
            glm::uvec2 indexBounds = startEndIndexes[currID];
            float density = 0.0f;
            for(int i = indexBounds.x; i < indexBounds.y; i++){

                float dist = glm::distance(currentP, particlesLocations[indexesByCellID[i]]);
                float q = dist/m_smoothingRadius;

                if (q <= 1){
                    density += smoothingQuadraticSpike(q);
                    counter.y++;
                }

            }
            
            densities[j].y = density;
            density = 0;
            
            for (int i = 0; i < particlesLocations.size(); i++){
                float dist = glm::distance(currentP, particlesLocations[i]);
                float q = dist/m_smoothingRadius;

                if (q <= 1){
                    density += smoothingQuadraticSpike(q);
                    counter.x++;
                }
            }

            densities[j].x = density;
        }

        std::cout << "NEIGHBOOR CPU COUNTER: " 
                    << ", ALL_D: " << counter.x << " , NEIGH_D: " << counter.y << std::endl; 

        std::cout << "----------------------" << std::endl;
    }

    void compareDensities(){

        std::vector<glm::vec2> tmp(plCount, glm::vec2(0));
        std::vector<glm::uvec2> counter (1, glm::uvec2(0));        
        std::vector<glm::vec2> CPUdensities (plCount, glm::vec2(0));        

        bufferDensities.getData(tmp, plCount);
        debugBufferCounter.getData(counter, 1);

        // Ensure CPU reads are complete before proceeding
        glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);

        calculateDensitiesCPU(CPUdensities);

        std::cout << "NEIGHBOOR GPU COUNTER: " 
                    << " ALL_D: " << counter[0].x << " , NEIGH_D: " << counter[0].y << std::endl; 

        std::cout << "----------------------" << std::endl;
        int count = 0;

        // for (int i = 0; i < plCount; i++){
        //     // if (abs(tmp[i].x - tmp[i].y) > 0.001){
        //     //     std::cout << "DENSITIES_GPU DIFFER INDEX: " << i 
        //     //         << ", ALL_D: " << tmp[i].x << " , NEIGH_D: " << tmp[i].y << std::endl; 
        //     // }
        //     if (abs(tmp[i].x - CPUdensities[i].x) > 0.001 || abs(tmp[i].x - CPUdensities[i].y) > 0.001){
        //         std::cout << "DENSITIES INDEX: " << i 
        //             << ", ALL_GPU_D: " << tmp[i].x << " , ALL_CPU_D: " << CPUdensities[i].x <<
        //             "   -------   N_GPU: " << tmp[i].x << ", NEIGH_CPU_D: " << CPUdensities[i].y << std::endl; 
        //         count++;
        //     }
            
        // }
        std::cout << "Correct: " << plCount - count << ", False: " << count << std::endl;
    }

    void verifyNeighbooringArrays(){

        const int maxID = (gridNum.y - 1) * gridNum.x + gridNum.x;

        for (int i = 0; i < maxID; i++){
            auto indexes = startEndIndexes[i];
            auto dictIndexes = IDGridDict[i];
            int count = 0;
            for (int j = indexes.x; j < indexes.y; j++)
            {
                if (dictIndexes[count] != indexesByCellID[j])
                    std::cout << "differnt Index, IDGridDict:" << dictIndexes[count]
                        << ", indexesByCellID: " << indexesByCellID[j] << std::endl;
                count++;
            }
        }

    }


    // -------------------------------------

    // FUNCTIONS FOR SPATIAL HASHING

    // -------------------------------------


    void initSpatialHash(){

        spatialLookUp.clear();
        spatialOffsets.clear();
        spatialLookUp.resize(plCount);
        spatialOffsets.resize(plCount);

        for(int i = 0; i < plCount; i++)
            spatialLookUp[i] = {getCellKeyByIndex(i), i};

        std::sort(spatialLookUp.begin(), spatialLookUp.end(), &Solver::compareSpatialLookUp);

        for(int i = 0; i < plCount; i++){
        
            int key = spatialLookUp[i].x;
            int prevKey = (i == 0 ? plCount + 1 : spatialLookUp[i - 1].x);

            if (key != prevKey){

                spatialOffsets[key] = i;
            }

        }
        

    }

    void getNeighboorsSH(const glm::vec2& pointLocation, std::vector<int>& out) {

        std::lock_guard<std::mutex> lock(neighborsMutex);

        int ID = getID(pointLocation);

        for(int i = 0; i < 9; i++){
            
            int currCellKey = getCellKey(ID + neighbooringIDs[i]);
            int startIndex = spatialOffsets[currCellKey];

            for(int j = startIndex; j < plCount; j++){
                if (spatialLookUp[j].x != currCellKey) break;
                out.push_back(spatialLookUp[j].y);
            }
        }


    } 

    int getCellKeyByIndex(int particleIndex){
        return (getID(particlesLocations[particleIndex]) * 7919) % plCount;
    }

    int getCellKey(int ID) {
        return (ID * 7919) % plCount;
    }

    static bool compareSpatialLookUp(glm::ivec2 i1, glm::ivec2 i2){
        return i1.x < i2.x;
    }


    // -------------------------------------

    // FUNCTIONS FOR SOLVE

    // -------------------------------------

    void parrallel(float dt){

        if(plCount == 0) return;

        const int workCount = (int)plCount > 100 ? plCount/totalThreads : 0;
        dt = 0.05;
    
        updatePreviousLocations();
        initSpatialHash();
        
        // COMPUTING LOCAL AND NEAR DENSITY
        for (int i = 0; i < totalThreads && workCount; i++){
            int start = i * workCount;
            int end = (i == totalThreads - 1) ? plCount : workCount + start;
            threads.emplace_back(&Solver::computeNearALocalDensitiesParallel, this, start, end);
        }
        joinThreads(workCount);
        


        // APPLYING PRESSURE
        for (int i = 0; i < totalThreads && workCount; i++){
            int start = i * workCount;
            int end = (i == totalThreads - 1) ? plCount : workCount + start;
            threads.emplace_back(&Solver::applyPressureParallel, this, dt, start, end);
        }
        joinThreads(workCount);


        // APPLYING VELOCITIES AND GRAVITY
        for (int i = 0; i < totalThreads && workCount; i++){
            int start = i * workCount;
            int end = (i == totalThreads - 1) ? plCount : workCount + start;
            threads.emplace_back(&Solver::applyVelocitiesNGravityParallel, this, dt, start, end);
        }
        joinThreads(workCount);

        
        // BOUNDS CHECKING
        for (int i = 0; i < totalThreads && workCount; i++){
            int start = i * workCount;
            int end = (i == totalThreads - 1) ? plCount : workCount + start;
            threads.emplace_back(&Solver::borderResolutionParallel, this, start, end);
        }
        joinThreads(workCount);

        for(int i = 0; i < plCount; i++)
            if (particlesLocations[i].y < 0)
                assert(false);

        particlesLocationBuffer.setData(particlesLocations.data(), particlesLocations.size() * sizeof(glm::vec2));

    }

    void applyPressureParallel(float dt, int start, int end){

        for(int i = start; i < end; i++){
            
            glm::vec2& currentP = particlesLocations[i];

            // Pressure
            float P = pressureMultiplier * (particlesDensities[i] - targetDensity);
            // Near Pressure
            float Pnear =  pressureNearMultiplier * particlesNearDensities[i];
            auto& neighboorIndexes = getNeighboors(currentP);

            glm::vec2 dx = {0.f, 0.f};

            for (int j = 0; j < neighboorIndexes.size(); j++){

                float distance = glm::distance(currentP, particlesLocations[neighboorIndexes[j]]);
                glm::vec2 unit_R = currentP - particlesLocations[neighboorIndexes[j]];

                float q = distance/m_smoothingRadius;
    
                if ( q > 1 || (unit_R.x == 0 && unit_R.y == 0)) continue;

                unit_R = unit_R / glm::length(unit_R);
                glm::vec2 D = (unit_R * (P * smoothingQuadraticSpikeDerivative(q) + Pnear * smoothingQuadraticSpike(q)) * dt * dt);

                // Velocity difference
                glm::vec2 u = unit_R * (velocities[i] - velocities[neighboorIndexes[j]]);

                if (u.x > 0 && u.y > 0){
                    glm::vec2 I =  unit_R * smoothingQuadraticSpikeDerivative(q) * (sigma * u + beta * u * u) * dt;
                    
                    velocities[i] -= I/2.f;

                    velocities[neighboorIndexes[j]] += I/2.f;

                }


                particlesLocations[neighboorIndexes[j]] -= D/2.f;
                dx += D/2.f;
            }

            currentP += dx;
            
        }


    }


    void computeNearALocalDensitiesParallel(int start, int end){

        for(int i = start; i < end; i++){
            particlesDensities[i] = calculateDensity(i);
            particlesNearDensities[i] = calculateNearDensity(i);
        }
    }


    float calculateDensity(int i){

        float density = 0;
        // static int mass = 1;
        float distance = 0;

        neighboorCounter = 1;

        auto start = std::chrono::high_resolution_clock::now();
        ////////////
        std::vector<int> neighboors = getNeighboors(particlesLocations[i]);
        // getNeighboors(particlesLocations[i]);

        auto end = std::chrono::high_resolution_clock::now();
        dictionaryHashNeigh += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


        start = std::chrono::high_resolution_clock::now();
        //////////////
        // std::shared_ptr<std::vector<int>> neighboors = getNeighboorsSH(particlesLocations[i]);
        std::vector<int> out;
        getNeighboorsSH(particlesLocations[i], out);
        end = std::chrono::high_resolution_clock::now();
        spatialHashGetNeigh += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        
        for (auto index: neighboors){
            distance = glm::distance(particlesLocations[i], particlesLocations[index]);    
            // Calculating local particle density
            density += smoothingQuadraticSpike(distance/m_smoothingRadius);
        }
        return density;
    }


    float calculateNearDensity(int i){

        float density = 0;
        // const static int mass = 1;
        float distance = 0;

        std::vector<int>& neighboors = getNeighboors(particlesLocations[i]);
        // std::shared_ptr<std::vector<int>> neighboors = getNeighboorsSH(particlesLocations[i]);
        
        for (auto index: neighboors){
            distance = glm::distance(particlesLocations[i], particlesLocations[index]);    
            // Calculating Near particle density
            density += nearDensityKernel(distance/m_smoothingRadius);
        }
        return density;
    }


    void applyVelocitiesNGravityParallel(float dt, int start, int end){
        
        for(int i = start; i < end; i++){
                                                                                                    
            velocities[i] += (gravity + particlesLocations[i] - particlesPrevLocations[i]) * dt ;
            particlesLocations[i] += velocities[i]/dt;
        }

    }


    void borderResolutionParallel(int start, int end){

        static float collisionDamping = 0.4;
        static float borderFriction = 0.8;

        for(int i = start; i < end; i++)
        {
            glm::vec2& p = particlesLocations[i];
            // glm::vec2& prev = particlesPrevLocations[i];


            if(p.x < m_borderW.x){
                p.x = m_borderW.x + 1;
                velocities[i].x *= -collisionDamping;
                // velocities[i] += -velocities[i] * borderFriction;
            }
            else if (p.x > m_borderW.y){
                p.x = m_borderW.y - 1;
                velocities[i].x *= -collisionDamping;
                // velocities[i] += -velocities[i] * borderFriction;
            }

            if(p.y < m_borderH.x){
                p.y = m_borderH.x + 1;
                velocities[i].y *= -collisionDamping;
                // velocities[i] += -velocities[i].x * borderFriction;
            }
            else if (p.y > m_borderH.y){
                p.y = m_borderH.y - 1;
                velocities[i].y *= -collisionDamping;
                // velocities[i] += -velocities[i].x * borderFriction;
            }


            
            // float qD = (p.y - m_borderH.x)/m_smoothingRadius;
            // if(qD < 1){
            //     velocities[i].y += (1-qD) * 0.3;  
            // }
            
        }
    }


    // ------------------------------------------------------------------------------- //

    // KERNEL FUNCTIONS BELLOW

    // ------------------------------------------------------------------------------- // 


    float smoothingCubicSplineKernel(float q){

        if (q <= 0.5){
            return ((6 * ((q * q * q) - (q * q))) + 1);
        }
        else if (q <= 1){
            return (2 * (1 - q) * (1 - q)*  (1 - q));
        }
        else 
            return 0;
    }

    float smoothingQuadraticSpike(float q){
        
        if (q <= 1)
            return (1 - q) * (1 - q);
        return 0;
    }

    float nearDensityKernel(float q){
        
        if (q <= 1)
            return (1 - q) * (1 - q) * (1 - q);
        return 0;
    }

    float smoothingQuadraticSpikeDerivative(float q){
        
        if (q <= 1)
            return (1 - q);
        return 0;
    }


   

    // ------------------------------------------------------------------------------- //




    // HELPER FUNCTIONS BELLOW




    // ------------------------------------------------------------------------------- // 


    void calculateGrid(){

        
        gridNum.x = std::ceil((m_borderW.y - m_borderW.x)/m_smoothingRadius);
        gridNum.y = std::ceil((m_borderH.y - m_borderH.x)/m_smoothingRadius);

        neighbooringIDs[0] = 1;                   // right
        neighbooringIDs[1] = -1;                  // left
        neighbooringIDs[2] = gridNum.x;           // up
        neighbooringIDs[3] = gridNum.x + 1;       // up right
        neighbooringIDs[4] = gridNum.x - 1;       // up left
        neighbooringIDs[5] = -gridNum.x;          // down
        neighbooringIDs[6] = -gridNum.x + 1;      // down right
        neighbooringIDs[7] = -gridNum.x - 1;      // down left
        neighbooringIDs[8] = 0;                   // origin
    }

    void clearParticles(){
        particlesLocations.clear();
        particlesLocations.shrink_to_fit();
        particlesDensities.clear();
        particlesDensities.shrink_to_fit();
        particlesNearDensities.clear();
        particlesNearDensities.shrink_to_fit();
        plSize = 0;
        plCount = 0;
        clearBuffers();
    }

    void clearBuffers(){
        particlesLocationBuffer.clear();
    }

    void resetCache(){

        calculateGrid();

        cacheParticles();

        // computeShadersCalcuateNeighboors();

        initSpatialHash();

    }

    void cacheParticles(){

        pointGridDict.clear();
        IDGridDict.clear();
        
        
        for (int i = 0; i < plCount; i++){
            pointGridDict[getID(particlesLocations[i])].push_back(i);
        }

    }

    void initilizePrevPos(){
        // assert(particlesPrevLocations.empty());
        particlesPrevLocations.clear();
        particlesPrevLocations.shrink_to_fit();
        particlesPrevLocations.resize(plCount);
        
        for(int i = 0; i < plCount; i++)
            particlesPrevLocations[i] = (particlesLocations[i] - initialVelocity);
    }

    void updatePreviousLocations(){
        
        particlesPrevLocations = particlesLocations;
    }

    void joinThreads(bool flag){

        for (int i = 0; i < totalThreads && flag; i++){
            if (threads[i].joinable()) {
                threads[i].join();
            }
        }
        threads.clear();

    }

    void toggleAttraction(bool flag, glm::vec2 mouseP){
        isPressed = flag;
        mousePos = mouseP;
    }
    
    // ------------------------------------------------------------------------------- //



    // SETTER FUNCTIONS BELLOW



    // ------------------------------------------------------------------------------- // 

    void setParticlesLocation(std::vector<glm::vec2> IN_particlesLocations){

        particlesLocations.clear();
        particlesDensities.clear();
        particlesPredictedLocations.clear();
        particlesNearDensities.clear();
        velocities.clear();

        particlesLocations = IN_particlesLocations;
        plSize = sizeof(glm::vec2) * IN_particlesLocations.size();
        plCount = IN_particlesLocations.size();

        particlesLocationBuffer.setData(particlesLocations.data(), plCount * sizeof(glm::vec2));


        velocities.resize(plCount);
        particlesDensities.resize(plCount);
        particlesNearDensities.resize(plCount);
        particlesPredictedLocations.resize(plCount);

        // INITIALIZING GPU BUFFERS
        initilizePrevPos();
        computeShadersCalcuateNeighboors();
        std::vector<glm::vec2> initialVelocities(plCount, {0, 0});

        bufferVelocities.setData(initialVelocities.data(), plCount * sizeof(glm::vec2));
        bufferPrevVelocities.setData(initialVelocities.data(), plCount * sizeof(glm::vec2));
        bufferPrevPositions.setData(IN_particlesLocations.data(), plCount * sizeof(glm::vec2));
        bufferDensities.setData(initialVelocities.data(), plCount * sizeof(glm::vec2));
        // debugPressure.setData(initialVelocities.data(), plCount * sizeof(glm::vec4));


        glm::ivec2 tmp {0, 0};
        // debugBufferCounter.setData(&tmp, sizeof(glm::ivec2));

        std::vector<glm::ivec2> a (plCount, tmp);
        spatialLookupBuffer.setData(a.data(), plCount * sizeof(glm::ivec2));

        std::vector<int> b (plCount, 0);
        spatialOffsetsBuffer.setData(b.data(), plCount * sizeof(int));
        // debugID.setData(b.data(), plCount * sizeof(int));

    }

    void setSmoothingRadius(float radius){
        assert(radius >= 1);
        m_smoothingRadius = radius;
        diameter = 2 * radius;
        resetCache();
    }

    void setBorder(float border[]){

        m_borderW.x = border[0];
        m_borderW.y = border[1];
        m_borderH.x = border[2];
        m_borderH.y = border[3];

        resetCache();
    }

    void setTargetDensity(float targetDens){
        targetDensity = targetDens;
    }

    void setPressureMultiplier(float pressureMult){
        pressureMultiplier = pressureMult;
    }

    void setPressureNearMultiplier(float pressureNearMult){
        pressureNearMultiplier = pressureNearMult;
    }

    void setGravity(glm::vec2 g){
        gravity = g;
    }

    void setSigma(float s){
        sigma = s;
    }

    void setBeta(float b){
        beta = b;
    }

    void setAttraction(float a){
        attraction = a;
    }
    // ------------------------------------------------------------------------------- //



    // GETTER FUNCTIONS BELLOW



    // ------------------------------------------------------------------------------- // 
    
    
    std::vector<int>& getNeighboors(const glm::vec2& pointLocation){

        std::lock_guard<std::mutex> lock(neighborsMutex);

        int ID = getID(pointLocation);
        
        if (IDGridDict.find(ID) != IDGridDict.end())
            return IDGridDict[ID];

        for(int i = 0; i < 9; i++){

            if (pointGridDict.find(ID + neighbooringIDs[i]) != pointGridDict.end())
                for (auto& point: pointGridDict[ID + neighbooringIDs[i]])
                    IDGridDict[ID].push_back(point);

        }
        return IDGridDict[ID];

    }

    std::vector<int>& getNeighboorsByID(int ID){

        std::lock_guard<std::mutex> lock(neighborsMutex);
        
        if (IDGridDict.find(ID) != IDGridDict.end())
            return IDGridDict[ID];

        for(int i = 0; i < 9; i++){

            if (pointGridDict.find(ID + neighbooringIDs[i]) != pointGridDict.end())
                for (auto& point: pointGridDict[ID + neighbooringIDs[i]])
                    IDGridDict[ID].push_back(point);

        }
        return IDGridDict[ID];

    }

    void computeShadersCalcuateNeighboors(){

        const int maxID = (gridNum.y - 1) * gridNum.x + gridNum.x;

        const int workCount = (int)maxID > 6 ? maxID/totalThreads : 1;

        startEndIndexes.clear();
        startEndIndexes.resize(maxID);
        indexesByCellID.clear();
        indexesByCellID.shrink_to_fit();


        // if (workCount != 1)
        // {
        //     for (int i = 0; i < totalThreads; i++){
        //         int start = i * workCount;
        //         int end = (i == totalThreads - 1) ? maxID : workCount + start;
        //         threads.emplace_back(
        //             [this, start, end]() {
        //                 for (int ID = start; ID < end; ID++) {
        //                     getNeighboorsByID(ID);
        //                 }
        //             }
        //         );
        //     }
        //     joinThreads(workCount);
        // }
        // else
        //     for (int ID = 0; ID < maxID; ID++) 
        //         getNeighboorsByID(ID);
        cacheParticles();
            

        int currIndex = 0;

        for(int ID = 0; ID < maxID; ID++){
            startEndIndexes[ID].x = currIndex;
            std::vector<int> neighboorIndexes = getNeighboorsByID(ID);

            for(int index: neighboorIndexes){
                indexesByCellID.push_back(index);
                currIndex++;
            }
            startEndIndexes[ID].y = currIndex;
        }



    }

    void updateNearBorderParticles(){

        nearBorderParticlesIndex.clear();
        nearBorderParticlesIndex.shrink_to_fit();

        for(int id = 0; id < gridNum.x; id++){
            // Particles of first row
            if (pointGridDict.find(id) != pointGridDict.end())
                for(auto index: pointGridDict[id])
                    nearBorderParticlesIndex.push_back(index);

            int lastRowID = id + (gridNum.y - 1) * gridNum.x;
            // Particles of last row
            if (pointGridDict.find(lastRowID) != pointGridDict.end())
                for(auto index: pointGridDict[lastRowID])
                    nearBorderParticlesIndex.push_back(index);
        }

        for(int id = 1; id < gridNum.y; id++){
            
            int leftSideID = id * gridNum.x;

            if (pointGridDict.find(leftSideID) != pointGridDict.end())
                for(auto index: pointGridDict[leftSideID])
                    nearBorderParticlesIndex.push_back(index);
            
            int rightSideID = (id + 1) * gridNum.x - 1;

            if (pointGridDict.find(rightSideID) != pointGridDict.end())
                for(auto index: pointGridDict[rightSideID])
                    nearBorderParticlesIndex.push_back(index);

        } 

      

    }


    std::vector<glm::vec2> getParticlesLocations() const{
        return particlesLocations;
    }

    glm::vec2* getParticlesLocationsData() {
        return particlesLocations.data();
    }

    unsigned int getParticlesLocationsSize() const {
        return plSize;
    }

    unsigned int getParticlesLocationCount() const{
        return plCount;
    }

    float getSmoothingRadius() const { 
        return m_smoothingRadius;
    }

    float getAttractionForce(){
        return attraction;
    }

    float getSigma(){
        return sigma;
    }

    float getBeta(){
        return beta;
    }
    
    int getID(const glm::vec2& particleLocation) const{
        // int row = (particleLocation.y < m_borderH.x) ? gridNum.x * (std::ceil((particleLocation.y - m_borderH.x)/(m_smoothingRadius * 2)) - 1) : 0;
        // int col = (particleLocation.x < m_borderW.x) ? (std::ceil((particleLocation.x - m_borderW.x)/(m_smoothingRadius * 2)) - 1) : 0;

        // return row + col;
        return gridNum.x * (std::ceil((particleLocation.y - m_borderH.x)/m_smoothingRadius) - 1) + (std::ceil((particleLocation.x - m_borderW.x)/m_smoothingRadius) - 1);
    }

    float* getDensityData() {
        return particlesDensities.data();
    }

    // Returns size of densities array in bytes
    unsigned int getDensitySize() const {
        return particlesDensities.size() * sizeof(float);
    }

    float getTargetDensity() const {
        return targetDensity;
    }
    
    float getPressureMultiplier() const {
        return pressureMultiplier;
    }

    float getPressureNearMultiplier() const {
        return pressureNearMultiplier;
    }

    glm::vec2 getGravity(){
        return gravity;
    }

    int* getNeighbooringOffesets(){
        return neighbooringIDs;
    }

    glm::i16vec2 getGridNum(){
        return gridNum;
    }
};