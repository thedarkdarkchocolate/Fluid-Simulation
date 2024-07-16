#pragma once
#include <glad/glad.h>



template <typename Type> class ShaderBuffer {

    unsigned int ID;
    unsigned int bind;

public:

    ShaderBuffer(){}

    ShaderBuffer(const void* data, unsigned int size, const int binding = 0){

        glGenBuffers(1, &ID);

        bind = binding;

        Bind();

        glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, GL_DYNAMIC_COPY);

        BindBufferBase(binding);

        Unbind();

    }

    ShaderBuffer(int binding){

        glGenBuffers(1, &ID);
        
        bind = binding;

        Bind();

        BindBufferBase(binding);

        Unbind();
    }
    
    void setData(const void* data, unsigned int size){

        Bind();

        glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, GL_DYNAMIC_COPY);

        BindBufferBase(bind);
        
        Unbind();
    }

    
    void getData(std::vector<Type>& outData, int count){

        Bind();

        Type* array = (Type*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, count * sizeof(Type), GL_MAP_READ_BIT);

        for(int i = 0; i < count; i++){
            outData[i] = array[i];  
        }

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        Unbind();

    }

    void clear(){

        Bind();

        glBufferData(GL_SHADER_STORAGE_BUFFER, 1, nullptr, GL_DYNAMIC_COPY);

        Unbind();
    }


    void BindBufferBase(const int binding){
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ID);
    }
    
    void Bind(){
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ID);
    }

    void Unbind(){
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }


};
