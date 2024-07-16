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


layout(std430, binding = 0) readonly buffer particlesLocations {
    vec2 particles[];
};

layout(std430, binding = 1) readonly buffer particlesByCellID {
    // This will hold all the particles indexes by id and a seperate array will hold the start and end indexes of every cell ID 
    uint indexesByCellID[];
};

layout(std430, binding = 2) readonly buffer neighbooringParticlesIndex {
    // This will hold the start and end indexes for neihbooring particles indexes for particlesLocations
    uvec2 startEndIndexes[];
};

int getID(int index){
    return int(gridNum.x * (ceil((particles[index].y - border.z)/(smoothingRadius * 2)) - 1) + (ceil((particles[index].x - border.x)/(smoothingRadius * 2)) - 1));
}


void main () {
    
    // gl_FragColor += vec4((cos(gl_FragCoord.y/225 * 3.1415 - 3 + sin(gl_FragCoord.x/900 * 3.1415 * 4)) + 1)/2);
    
    // ivec2 indexBounds = ivec2(startEndIndexes[currID]);

    // for(int i = indexBounds.x; i < indexBounds.y; i++){

    for (int i = 0; i < particles.length(); i++){

        float dist = distance(gl_FragCoord.xy, particles[i].xy);

        // Drawing solid particles
        if (dist <= 2) gl_FragColor = vec4(1.f);

        if (dist <= smoothingRadius){
            
            float q = dist/smoothingRadius;
            float outColor = 0;

            // Dimension normalization factors s2 = 40/((7 * 3.1415 * smoothingRadius * smoothingRadius)), s3 =  8/((3.1415 * smoothingRadius * smoothingRadius * smoothingRadius))
            if (q <= 0.5)
                outColor = ((6 * ((q * q * q) - (q * q))) + 1);
            else if (q <= 1)
                outColor = (2 * (1 - q) * (1 - q) * (1 - q));
            else   
                outColor = 0;

            gl_FragColor += vec4(0, 0, outColor * 0.1, 0);

        }

    }      
    

    // int ID = getID(particlesIndex);
    // ivec2 indexBounds = ivec2(startEndIndexes[ID]);

    // for(int k = indexBounds.x; k < indexBounds.y; k++){

    //     float dist = distance(gl_FragCoord.xy, particles[indexesByCellID[k]]);

    //     // Drawing solid particles
    //     if (dist <= 2) gl_FragColor = vec4(1.f, 0, 0, 1);
    //     // if ((dist/smoothingRadius) <= 1 && dist <= 2)
    //         // gl_FragColor += vec4(0, 0, 1.f, 1);
    // }    

    // for (int i = 0; i < particles.length(); i++){
    //     if (i == 1) continue;
    //     float dist = distance(gl_FragCoord.xy, particles[i]);
    //     // float dist2 = distance(particles[1], particles[i]);
    //     // if ((dist/smoothingRadius) <= 1 && dist2/smoothingRadius < 1 ){
    //     if ((dist/smoothingRadius) <= 1 ){
    //         // Drawing solid particles
    //         if (dist <= 2) gl_FragColor += vec4(0, 0, 1.f, 1);

    //     }
    // }

    // if (distance(gl_FragCoord.xy, particles[particlesIndex]) < 2.f)
    //     gl_FragColor = vec4(0, 1.f, 0, 1);
        
    if (drawGrid)
        if (gl_FragCoord.x >= border.x && gl_FragCoord.x <= border.y && gl_FragCoord.y >= border.z && gl_FragCoord.y <= border.w) 
            if (mod(gl_FragCoord.x - border.x , smoothingRadius * 2.0) < 1.0 || mod(gl_FragCoord.y - border.z, smoothingRadius * 2.0) < 1.0)
                gl_FragColor = vec4(0.31);
    
    if (isPressed && distance(gl_FragCoord.xy, densityCenter) <= smoothingRadius + 1 && distance(gl_FragCoord.xy, densityCenter) >= smoothingRadius - 1){
            gl_FragColor += vec4(1.f);

    }

}