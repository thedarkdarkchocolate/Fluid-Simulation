#version 430

uniform float smoothingRadius;


void main() {

    gl_FragColor = vec4(0.f, .6f, 0.f, 1.f);
   
    // if (mod(outPos.x, smoothingRadius) < 1)
    //     gl_FragColor = vec4(1);

}