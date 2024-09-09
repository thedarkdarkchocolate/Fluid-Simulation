#pragma once 

#include <glad/glad.h>
#include <glm\gtc\type_ptr.hpp>


#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

class Shader {

    // program ID
    unsigned int ID;
    
    mutable std::unordered_map<std::string, GLint> uniformLocationsCache;

    public:
    // constructor reads and builds the shader program
    Shader(const char * vertexPath, const char* fragmentPath);
    Shader(const std::string vertexPath, const std::string fragmentPath);

    // Copy constuctor
    // Shader(Shader& shader) = delete;

    // Shader(Shader&& shader) = delete;

    ~Shader();

    // Use/Activate Shader
    void useShader();

    // Get program ID
    unsigned int getProgramID();

    // Utility uniform functions
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setUint(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2fv(const std::string& name, glm::vec2 vec2) const;
    void setVec3f(const std::string& name, glm::vec3 vec3) const;
    void setVec3fv(const std::string& name, glm::vec3 vec3) const;
    void setVec4fv(const std::string& name, glm::vec4 vec4) const;
    void setVec4f(const std::string& name, glm::vec4 vec4) const;
    void setUniformMat4(const std::string &name, glm::mat4& model) const;
    void setIntArray(const std::string &name, int value[], int size) const;
    
    GLint getUniformLocation(unsigned int ID, const std::string& name) const;


};
