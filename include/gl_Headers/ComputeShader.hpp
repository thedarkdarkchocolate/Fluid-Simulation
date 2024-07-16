#pragma once 
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>


class ComputeShader 
{

    unsigned int ID;
    mutable std::unordered_map<std::string, GLint> uniformLocationsCache; 

    public:
    // Constructor for Compute Shaders
    ComputeShader(const char* computePath) {


        std::string computeCode;
        std::ifstream cShaderFile;

        cShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);

        try{
            
            // Code for reading shaders from another file 
            // Opening files 
            cShaderFile.open(computePath);

            // Declaring streams
            std::stringstream cShaderStream;

            // read file's buffer content into streams
            cShaderStream << cShaderFile.rdbuf();

            // Closing opened files
            cShaderFile.close();

            // Convert stream into string
            computeCode  = cShaderStream.str();

        }catch(std::ifstream::failure e)
        {
            std::cout << "ERROR::COMPUTES_SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
        }


        const char * cShaderCode = computeCode.c_str();


        // Compile Shader
        unsigned compute;
        int success;
        char infoLog[512];

        // Compute Shader
        compute = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(compute, 1, &cShaderCode, NULL);
        glCompileShader(compute);   
        
        glGetShaderiv(compute, GL_COMPILE_STATUS, &success);
        
        if(!success){

            glGetShaderInfoLog(compute, 512, NULL, infoLog);
            std::cout << "ERROR::COMPUTES_SHADER::COMPUTE::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        
        // Creating program
        ID = glCreateProgram();
        glAttachShader(ID, compute);
        glLinkProgram(ID);

        // Program linking error check
        glGetProgramiv(ID, GL_LINK_STATUS, &success);

        if(!success){

            glGetProgramInfoLog(ID, 512, NULL, infoLog);
            std::cout << "ERROR::COMPUTES_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        
        glDeleteShader(compute);


    }

    ~ComputeShader(){};

    void useShader()
    {
        glUseProgram(ID);
    }

    unsigned int getProgramID()
    {
        return ID;
    }

    void setBool(const std::string &name, bool value) const
    {    
        glUniform1i(getUniformLocation(ID, name), (int)value); 
    }

    void setVec4fv(const std::string& name, glm::vec4 vec4) const
    {
        glUniform4fv(getUniformLocation(ID, name), 1, &vec4[0]);
    }

    void setVec4f(const std::string& name, glm::vec4 vec4) const
    {
        glUniform4f(getUniformLocation(ID, name), vec4.x, vec4.y, vec4.z, vec4.w);
    }

    void setVec3fv(const std::string& name, glm::vec3 vec3) const
    {
        glUniform3fv(getUniformLocation(ID, name), 1, &vec3[0]);
    }

    void setVec3f(const std::string& name, glm::vec3 vec3) const
    {
        glUniform3f(getUniformLocation(ID, name), vec3.x, vec3.y, vec3.z);
    }

    void setVec2fv(const std::string& name, glm::vec2 vec2) const
    {
        glUniform2fv(getUniformLocation(ID, name), 1, &vec2[0]);
    }


    void setInt(const std::string &name, int value) const
    { 
        glUniform1i(getUniformLocation(ID, name), value); 
    }

    void setUint(const std::string &name, int value) const
    { 
        glUniform1ui(getUniformLocation(ID, name), value); 
    }

    void setFloat(const std::string &name, float value) const
    { 
        glUniform1f(getUniformLocation(ID, name), value); 
    } 

    void setUniformMat4(const std::string &name, glm::mat4& model) const
    { 
        glUniformMatrix4fv(getUniformLocation(ID, name), 1, GL_FALSE, glm::value_ptr(model));
    } 

    void setIntArray(const std::string &name, int value[], int size) const
    {
        glUniform1iv(getUniformLocation(ID, name), size, value);
    }

    GLint getUniformLocation(unsigned int ID, const std::string& name) const
    {
        // Checking if uniform location exists in cache
        if(uniformLocationsCache.find(name) != uniformLocationsCache.end())
            return uniformLocationsCache[name];
        
        // If it doesn't we update the dict and return it 
        uniformLocationsCache[name] = glGetUniformLocation(ID, name.c_str());

        return uniformLocationsCache[name];

    }
};