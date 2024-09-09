#version 430 core

uniform float smoothingRadius;
uniform float particleRadius;
uniform vec4 border;
uniform vec2 densityCenter;
uniform bool isPressed;
uniform bool drawGrid;
uniform int particlesIndex;
uniform int neighbooringIDs[9];
uniform vec2 gridNum;

int currID;
uint currIndex;

layout(std430, binding = 0) readonly buffer particlesLocations {
    vec2 particles[];
};

layout(std430, binding = 1) readonly buffer particlesByCellID {
    uint neighboors[];
};

layout(std430, binding = 2) readonly buffer neighbooringParticlesIndex {
    uvec2 startEndIndexes[];
};

layout(std430, binding = 3) buffer particlesVelocities {
    vec2 velocities[];
};

void getID() {
    currID = int((gridNum.x * ceil((gl_FragCoord.y - border.z) / smoothingRadius) - 1) + (ceil((gl_FragCoord.x - border.x) / smoothingRadius) - 1) - gridNum.x + 1);
}

// Array of 16 colors for smoother transitions
vec3 getColorFromVelocity(vec2 velocity) {
    float speed = length(velocity);
    float maxSpeed = 20.0; // Adjust based on max speed in the simulation
    float t = clamp(speed / maxSpeed, 0.0, 1.0);

    vec3 colorPalette[16] = vec3[](
        vec3(0.0, 0.0, 1.0),  // 0: Blue
        vec3(0.0, 0.5, 1.0),  // 1: Light Blue
        vec3(0.0, 1.0, 1.0),  // 2: Cyan
        vec3(0.0, 1.0, 0.8),  // 3: Cyan-Green
        vec3(0.0, 1.0, 0.4),  // 4: Greenish Cyan
        vec3(0.0, 1.0, 0.0),  // 5: Green
        vec3(0.5, 1.0, 0.0),  // 6: Yellow-Green
        vec3(0.8, 1.0, 0.0),  // 7: Yellowish-Green
        vec3(1.0, 1.0, 0.0),  // 8: Yellow
        vec3(1.0, 0.8, 0.0),  // 9: Yellow-Orange
        vec3(1.0, 0.6, 0.0),  // 10: Orange
        vec3(1.0, 0.4, 0.0),  // 11: Deep Orange
        vec3(1.0, 0.2, 0.0),  // 12: Reddish Orange
        vec3(1.0, 0.0, 0.0),  // 13: Red
        vec3(0.8, 0.0, 0.0),  // 14: Dark Red
        vec3(0.6, 0.0, 0.0)   // 15: Deep Dark Red
    );

    // Map velocity to color
    int colorIndex = int(t * 15.0); // Scale t to get index from 0 to 15
    return colorPalette[colorIndex];
}

void main() {
    getID();

    ivec2 indexBounds = ivec2(startEndIndexes[currID]);

    vec4 finalColor = vec4(0.0);
    float particleRadiusScreenSpace = 3.0; // Adjust size of colored particles on screen

    for (int i = 0; i < particles.length(); i++) {
        float dist = distance(gl_FragCoord.xy, particles[i].xy);

        // Only color the particle center within a small radius
        if (dist <= particleRadiusScreenSpace) {
            vec3 particleColor = getColorFromVelocity(velocities[i]);

            // Set a fixed alpha for the particle color
            float alpha = 1.0 - (dist / particleRadiusScreenSpace);

            // Add the particle color, weighted by distance
            finalColor += vec4(particleColor, alpha);
        }
    }

    // Normalize final color to avoid oversaturation
    if (finalColor.a > 0.0) {
        gl_FragColor = finalColor / finalColor.a; // Normalize by alpha
    } else {
        gl_FragColor = vec4(0.0);
    }

    // Draw grid if needed
    if (drawGrid) {
        if (gl_FragCoord.x >= border.x && gl_FragCoord.x <= border.y && gl_FragCoord.y >= border.z && gl_FragCoord.y <= border.w) {
            if (mod(gl_FragCoord.x - border.x, smoothingRadius) < 1.0 || mod(gl_FragCoord.y - border.z, smoothingRadius) < 1.0) {
                gl_FragColor = mix(gl_FragColor, vec4(0.31), 0.5); // Blend grid with particle color
            }
        }
    }

    // Draw interaction circle if pressed
    if (isPressed && distance(gl_FragCoord.xy, densityCenter) <= smoothingRadius + 1.0 && distance(gl_FragCoord.xy, densityCenter) >= smoothingRadius - 1.0) {
        gl_FragColor = mix(gl_FragColor, vec4(1.0), 0.5); // Blend interaction circle with particle color
    }
}