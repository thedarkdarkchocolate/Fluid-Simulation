#pragma once

#include <glad/glad.h>
#include <vector>


struct VertexBufferEl{ 
    unsigned int type;
    unsigned int count;
    unsigned char normalized;
    
    static unsigned int getSizeOfType(unsigned int type){

        switch (type){

            case GL_FLOAT:          return sizeof(float);
            case GL_UNSIGNED_INT:   return sizeof(unsigned int);
            case GL_UNSIGNED_BYTE:  return sizeof(char);
            default:                assert(false);
        };

        return 0;
    }
};



class VertexArray
{
private:
    
    unsigned int VAO;
    std::vector<VertexBufferEl> m_buffElements;
    unsigned int m_stride;
    VertexBuffer m_VBO;

public:

    VertexArray();
    ~VertexArray();

    void addLayout(unsigned int type, unsigned int count, unsigned char norm);
    void addBuffer(const VertexBuffer& vbo);
    void addBuffer(const VertexBuffer& vbo, unsigned int stride);
    void applyTranforamtions(Shader& shaderToSendUniform, const std::string& uniformName, const glm::vec3& modelPosition, const float rotation = 0,
                                const glm::vec3& rotationAxis = glm::vec3(1.f), const glm::vec3& scale = glm::vec3(1.f));

    void setData(const void* data, unsigned int size);

    void Bind() const;
    void Unbind() const;

    // private:
    
    void disableAttribArray() const;

};

VertexArray::VertexArray()
: m_stride{0}
{
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
}

void VertexArray::addLayout(unsigned int type, unsigned int count, unsigned char norm){

    m_buffElements.push_back({type, count, norm});
    m_stride += VertexBufferEl::getSizeOfType(type) * count;
}

void VertexArray::addBuffer(const VertexBuffer& vbo){

    
    assert(!m_buffElements.empty());

    vbo.Bind();
    unsigned int offset = 0;

    m_VBO = vbo;

    for(int i = 0; i < m_buffElements.size(); i++){

        const VertexBufferEl currElement = m_buffElements[i] ;
        glVertexAttribPointer(i, currElement.count, currElement.type, currElement.normalized,
                            m_stride, (void*)offset);

        glEnableVertexAttribArray(i);

        offset += currElement.count * VertexBufferEl::getSizeOfType(currElement.type); 
    }

    m_buffElements.clear();
    
}


void VertexArray::addBuffer(const VertexBuffer& vbo, unsigned int stride){

    
    assert(!m_buffElements.empty());
    assert(stride >= 0);

    
    vbo.Bind();
    unsigned int offset = 0;

    m_VBO = vbo;

    for(int i = 0; i < m_buffElements.size(); i++){

        const VertexBufferEl currElement = m_buffElements[i] ;
        glVertexAttribPointer(i, currElement.count, currElement.type, currElement.normalized,
                            stride, (const void*) offset);

        glEnableVertexAttribArray(i);

        offset += currElement.count * VertexBufferEl::getSizeOfType(currElement.type); 
    }

    vbo.Unbind();
    m_buffElements.clear();
    
}


void VertexArray::applyTranforamtions(Shader& shaderToSendUniform, const std::string& uniformName, const glm::vec3& modelPosition, const float rotation, const glm::vec3& rotationAxis,
                                         const glm::vec3& scale){

    // this->Bind(); // Binding VAO

    glm::mat4 model = glm::mat4(1.f);
    model = glm::translate(model, modelPosition);

    if (rotation)
        model = glm::rotate(model, glm::radians(rotation), rotationAxis);

    if (scale != glm::vec3(1.f))
        model = glm::scale(model, scale);
    
    if (uniformName != "")
        shaderToSendUniform.setUniformMat4(uniformName.c_str(), model);
}

void VertexArray::setData(const void* data, unsigned int size){

    Bind();
    m_VBO.setData(data, size);
    Unbind();
    
}

void VertexArray::Bind() const
{
    glBindVertexArray(VAO);
}

void VertexArray::Unbind() const
{
    glBindVertexArray(0);
    disableAttribArray();
}

void VertexArray::disableAttribArray() const
{
    for(int i = 0; i < m_buffElements.size(); i++)
        glDisableVertexAttribArray(i);
}



VertexArray::~VertexArray()
{
    glDeleteVertexArrays(1, &VAO);
}

