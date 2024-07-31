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
    // This will hold all the particles indexes by id and a seperate array will hold the start and end indexes of every cell ID 
    uint neighboors[];
};

layout(std430, binding = 2) readonly buffer neighbooringParticlesIndex {
    // This will hold the start and end indexes for neihbooring particles indexes for particlesLocations
    uvec2 startEndIndexes[];
};

    
void getID(){
    currID = int((gridNum.x * ceil((gl_FragCoord.y - border.z)/smoothingRadius) - 1) + (ceil((gl_FragCoord.x - border.x)/smoothingRadius) - 1) - gridNum.x + 1);
}



void main () {
    
    getID();
    
    if (currID < 0) return;
    
    ivec2 indexBounds = ivec2(startEndIndexes[currID]);

    for(int i = indexBounds.x; i < indexBounds.y; i++){

    // for (int i = 0; i < particles.length(); i++){

        float dist = distance(gl_FragCoord.xy, particles[neighboors[i]].xy);

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
            if (mod(gl_FragCoord.x - border.x , smoothingRadius) < 1.0 || mod(gl_FragCoord.y - border.z, smoothingRadius) < 1.0)
                gl_FragColor = vec4(0.31);
    
    if (isPressed && distance(gl_FragCoord.xy, densityCenter) <= smoothingRadius + 1 && distance(gl_FragCoord.xy, densityCenter) >= smoothingRadius - 1){
            gl_FragColor += vec4(1.f);

    }

}