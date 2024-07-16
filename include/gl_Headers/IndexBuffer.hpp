#pragma once
#include <glad/glad.h>

class IndexBuffer 
{
private:
    unsigned int m_buffer_ID;

public:

    IndexBuffer(const unsigned int* data, unsigned int size);
    ~IndexBuffer();

    void Bind() const;
    void Unbind() const;
    unsigned int getBufferID();



};

IndexBuffer::IndexBuffer(const unsigned int* data, unsigned int size)
{

    glGenBuffers(1, &m_buffer_ID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer_ID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

}

unsigned int IndexBuffer::getBufferID(){
    return m_buffer_ID;
}

void IndexBuffer::Bind() const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer_ID);
}

void IndexBuffer::Unbind()const
{
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

IndexBuffer::~IndexBuffer()
{
    glDeleteBuffers(1, &m_buffer_ID);
}

