#version 430 core


uniform float smoothingRadius;
uniform bool toggle;

layout(std430, binding = 0) readonly buffer particlesLocations {

    vec2 particles[];

};

layout(std430, binding = 1) readonly buffer example {

    float exampleProperty[];

};

void main(){

    bool kappa = false;

    for (int i = 0; i < particles.length(); i ++){


        float dist = distance(gl_FragCoord.xy, particles[i].xy);

        if (dist < 2.f){

            gl_FragColor = vec4(1, 0, 0, 0); 
            break;
        } 

        // gl_FragColor += vec4((smoothingRadius * smoothingRadius - dist * dist) /(smoothingRadius * smoothingRadius));
        // gl_FragColor = vec4((smoothingRadius - dist) / smoothingRadius );
        // gl_FragColor =

        // Epanechnikov Kernel Function
        // dist = dist/smoothingRadius;
        // gl_FragColor = vec4(0.75 * (1 - dist * dist));




        float q = dist/smoothingRadius;
        float outColor = 0;

        if (q <= 1){
            outColor = ((1 - q) * (1 - q));

        }

        // if (q <= 0.5)
        //     outColor = ((6 * ((q * q * q) - (q * q))) + 1);
        // else if (q <= 1)
        //     outColor = (2 * (1 - q) * (1 - q) * (1 - q));
        // else   
        //     outColor = 0;


        // if (q <= 1){
        //     float interpolation = (cos(particles[i].y/225 * 3.1415 - 3 + sin(particles[i].x/900 * 3.1415 * 4)) + 1)/2;
        //     float value = outColor * interpolation / exampleProperty[i];
        //     gl_FragColor += vec4(0.3 * value, 0.2 * value, 0.5 * value, 1) * vec4(1);
        // }
        // gl_FragColor += vec4(outColor, exampleProperty[i], outColor, 0);

        gl_FragColor += vec4(outColor * 0.1);
        // gl_FragColor += vec4(exampleProperty[i] * 0.01);
        // if (toggle){
        //     float exampleValue = (cos(gl_FragCoord.y/225 * 3.1415 - 3 + sin(gl_FragCoord.x/900 * 3.1415 * 4)) + 1)/2;
        //     gl_FragColor = vec4(0.3 * exampleValue, 0.2 * exampleValue, 0.5 * exampleValue, 1);
        // }
    }
    
    
    if (kappa)
        gl_FragColor = vec4(1.f, 0, 0, 0);

}