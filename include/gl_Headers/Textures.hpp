#pragma once

#include <string>
#include <glad/glad.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// static struct Material{
// };



class Texture{

    unsigned textureID;
    static int textureCounter;
    int textureN;

public:

    Texture (const char* path){
        
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        // set the texture wrapping/filtering options (on the currently bound texture object)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        int width, height, nrComponents;

        // Loading image
        unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);

        if(data){ 

            GLenum format;
            if (nrComponents == 1)
                format = GL_RED;
            else if (nrComponents == 3)
                format = GL_RGB;
            else if (nrComponents == 4)
                format = GL_RGBA;

            // Generating Texture
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            textureN = textureCounter;
            textureCounter++;

            // Freeing image 
            stbi_image_free(data);

        } else
        {   
            assert(false);
        }
        
    };

    //Empty Texture
    Texture(int WIDTH, int HEIGHT){

        glGenTextures(1, &textureID);

        BindTexture();

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGBA, 
                    GL_FLOAT, NULL);

        glBindImageTexture(0, textureID, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

        textureN = textureCounter;
        textureCounter++;
    }

    void BindTexture() const {
        glActiveTexture(GL_TEXTURE0 + textureN);
        glBindTexture(GL_TEXTURE_2D, textureID);
    }


    ~Texture() {};

};
int Texture::textureCounter = 0;