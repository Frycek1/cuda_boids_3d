#version 330 core

layout (location = 0) in vec3 aPos; 

layout (location = 1) in float iX;
layout (location = 2) in float iY;
layout (location = 3) in float iZ;

layout (location = 4) in float ivX;
layout (location = 5) in float ivY;
layout (location = 6) in float ivZ;

uniform mat4 view;
uniform mat4 projection;

out vec3 vColor;

void main()
{
    vec3 Pos = vec3(iX, iY, iZ);
    vec3 Vel = vec3(ivX, ivY, ivZ);


    
    vec3 Forward = normalize(Vel + vec3(0.00001));

    vec3 GlobalUp = vec3(0.0, 1.0, 0.0);
    
    if (abs(Forward.y) > 0.99) {
        GlobalUp = vec3(0.0, 0.0, 1.0);
    }

    vec3 Right = normalize(cross(Forward, GlobalUp)); // Wektor w prawo
    vec3 Up    = cross(Right, Forward);               // Wektor "grzbietu" boida

    mat4 rotation = mat4(1.0);
    
    rotation[0] = vec4(Right, 0.0);
    
    rotation[1] = vec4(Forward, 0.0);
    
    rotation[2] = vec4(Up, 0.0);

    mat4 translation = mat4(1.0);
    translation[3] = vec4(Pos, 1.0);

    // Kolejność: Projekcja * Kamera * Przesunięcie * Obrót * Geometria
    gl_Position = projection * view * translation * rotation * vec4(aPos, 1.0);

    float speed = length(Vel);
    float t = clamp((speed - 3.0) / 3.0, 0.0, 1.0); // Normalizacja 0..1 dla prędkości 3..6
    
    // Interpolacja koloru: Niebieski -> Cyjan -> Biały
    vColor = mix(vec3(0.1, 0.3, 0.8), vec3(0.6, 0.9, 1.0), t);
}