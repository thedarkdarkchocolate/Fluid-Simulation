#pragma once

#include <glad/glad.h>

class VertexBuffer
{

private:
    unsigned int VBO;

public:
    VertexBuffer() = default;
    VertexBuffer(const void* data, unsigned int size);
    VertexBuffer(VertexBuffer& other) = delete;
    ~VertexBuffer();

    void setData(const void* data, unsigned int size);
    void Bind() const;
    void Unbind() const;
    unsigned int getBufferID();

};



VertexBuffer::VertexBuffer(const void* data, unsigned int size)
{
    glGenBuffers(1, &VBO);
    Bind();
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

void VertexBuffer::setData(const void* data, unsigned int size){

    Bind();
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    
}

unsigned int VertexBuffer::getBufferID(){
    return VBO;
}

void VertexBuffer::Bind()const
{
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
}

void VertexBuffer::Unbind() const
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

VertexBuffer::~VertexBuffer()
{
    glDeleteBuffers(1, &VBO);
}
