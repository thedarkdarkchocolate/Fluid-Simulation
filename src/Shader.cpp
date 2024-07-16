#pragma once

#include "Shader.hpp"


// Constructor for Vertex and Fragment Shader
Shader::Shader(const char* vertexPath, const char* fragmentPath){


    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;    
    std::ifstream fShaderFile;

    vShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);

    try{
        
        // Code for reading shaders from another file 
        // Opening files 
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);

        // Declaring streams
        std::stringstream vShaderStream, fShaderStream;

        // read file's buffer content into streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();

        // Closing opened files
        vShaderFile.close();
        fShaderFile.close();

        // Convert stream into string
        vertexCode   = vShaderStream.str();
        fragmentCode = fShaderStream.str();


    }catch(std::ifstream::failure e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }


    const char* vShaderCode = vertexCode.c_str();
    const char * fShaderCode = fragmentCode.c_str();


    // Compile Shader
    unsigned vertex, fragment;
    int success;
    char infoLog[512];

    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);

    // Error checking
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);

    if(!success){

        glGetShaderInfoLog(vertex, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);

    // Error checking
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);

    if(!success){

        glGetShaderInfoLog(fragment, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Creating program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);

    // Program linking error check
    glGetProgramiv(ID, GL_LINK_STATUS, &success);

    if(!success){

        glGetProgramInfoLog(ID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertex);
    glDeleteShader(fragment);


}


// Constructor for Compute Shaders
Shader::Shader(const char* computePath){


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
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
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
        std::cout << "ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Creating program
    ID = glCreateProgram();
    glAttachShader(ID, compute);
    glLinkProgram(ID);

    // Program linking error check
    glGetProgramiv(ID, GL_LINK_STATUS, &success);

    if(!success){

        glGetProgramInfoLog(ID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(compute);


}

Shader::~Shader()
{
    glDeleteProgram(ID);
};

void Shader::useShader()
{
    glUseProgram(ID);
}

unsigned int Shader::getProgramID()
{
    return ID;
}

void Shader::setBool(const std::string &name, bool value) const
{    
    glUniform1i(getUniformLocation(ID, name), (int)value); 
    
}

void Shader::setVec3fv(const std::string& name, glm::vec3 vec3) const
{
    glUniform3fv(getUniformLocation(ID, name), 1, &vec3[0]);
}

void Shader::setVec2fv(const std::string& name, glm::vec2 vec2) const
{
    glUniform2fv(getUniformLocation(ID, name), 1, &vec2[0]);
}

void Shader::setVec3f(const std::string& name, glm::vec3 vec3) const
{
    glUniform3f(getUniformLocation(ID, name), vec3.x, vec3.y, vec3.z);
}

void Shader::setVec4fv(const std::string& name, glm::vec4 vec4) const 
{
    glUniform4fv(getUniformLocation(ID, name), 1, &vec4[0]);
}

void Shader::setVec4f(const std::string& name, glm::vec4 vec4) const
{
    glUniform4f(getUniformLocation(ID, name), vec4.x, vec4.y, vec4.z, vec4.w);
}

void Shader::setInt(const std::string &name, int value) const
{ 
    glUniform1i(getUniformLocation(ID, name), value); 
}

void Shader::setUint(const std::string &name, int value) const
{ 
    glUniform1ui(getUniformLocation(ID, name), value); 
}

void Shader::setFloat(const std::string &name, float value) const
{ 
    glUniform1f(getUniformLocation(ID, name), value); 
} 

void Shader::setUniformMat4(const std::string &name, glm::mat4& model) const
{ 
    glUniformMatrix4fv(getUniformLocation(ID, name), 1, GL_FALSE, glm::value_ptr(model));
} 

void Shader::setIntArray(const std::string &name, int value[], int size) const
{
    glUniform1iv(getUniformLocation(ID, name), size, value);
}

GLint Shader::getUniformLocation(unsigned int ID, const std::string& name) const
{
    // Checking if uniform location exists in cache
    if(uniformLocationsCache.find(name) != uniformLocationsCache.end())
        return uniformLocationsCache[name];
    
    // If it doesn't we update the dict and return it 
    uniformLocationsCache[name] = glGetUniformLocation(ID, name.c_str());

    return uniformLocationsCache[name];

}