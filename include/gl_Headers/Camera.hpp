#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


// Default camera values
const float YAW        = -90.0f;
const float PITCH      =  0.0f;
const float FOV        =  60.0f;

enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};


class Camera {

    glm::vec3 cameraPos;
    glm::vec3 cameraDir;
    glm::vec3 cameraUp;
    
    glm::vec3 right;


    glm::mat4 view;

    float yaw;
    float pitch;
    float fov;

    const float MovementSpeed = 2.5f;
    const float MouseSensitivity = 0.1f;
    const float ZoomSensitivity = 1.5f;

public:

    Camera ()
    {

        this->cameraPos = glm::vec3(0.f, 0.f, 3.f);
        this->cameraDir = glm::vec3(0.f, 0.f, -1.f);
        this->cameraUp = glm::vec3(0.f, 1.f, 0.f);

        this->yaw = YAW;
        this->pitch = PITCH;
        this->fov = FOV;
        
        this->updateCameraVectors();

    };

    Camera (glm::vec3 pos, glm::vec3 dir, glm::vec3 up)
    : cameraPos {pos} , cameraDir {dir}, cameraUp {up}
    {

        this->yaw = YAW;
        this->pitch = PITCH;
        this->fov = FOV;

        this->updateCameraVectors();

    }


    glm::mat4 getViewMatrix(){
        return glm::lookAt(this->cameraPos, this->cameraPos + this->cameraDir, this->cameraUp);
    }

    glm::vec3 getPosition(){
        return this->cameraPos;
    }

    void ProcessMouseScroll(double offset){

        fov -= (float)offset * 1.5;

        if (fov < 1.0f)
            fov = 1.0f;
        if (fov > 120.0f)
            fov = 120.0f; 
    }

    void ProcessKeyboard(Camera_Movement direction, float deltaTime)
    {
        float velocity = MovementSpeed * deltaTime;
        if (direction == FORWARD)
            cameraPos += cameraDir * velocity;
        if (direction == BACKWARD)
            cameraPos -= cameraDir * velocity;
        if (direction == LEFT)
            cameraPos -= right * velocity;
        if (direction == RIGHT)
            cameraPos += right * velocity;
        if (direction == UP)
            cameraPos += cameraUp * velocity;
        if (direction == DOWN)
            cameraPos -= cameraUp * velocity;
    }

    void processMouseMovement(double offsetX, double offsetY){

        offsetX *= MouseSensitivity;
        offsetY *= MouseSensitivity; 

        yaw += offsetX;
        pitch -= offsetY;

        if(pitch > 89.0f)
            pitch =  89.0f;
        if(pitch < -89.0f)
            pitch = -89.0f;

        this->updateCameraVectors();
    }

    glm::vec3 getDirection() const {
        return this->cameraDir;
    }
    
    void updateCameraVectors(){

        glm::vec3 dir;
        dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        dir.y = sin(glm::radians(pitch));
        dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));


        right = glm::normalize(glm::cross(cameraDir, cameraUp));
        cameraDir = glm::normalize(dir);

    }


    float getFOV(){
        return this->fov;
    }

};