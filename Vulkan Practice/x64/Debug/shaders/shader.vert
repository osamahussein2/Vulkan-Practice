#version 450

// Pass these per-vertex colors to the fragment shader so it can output their interpolated values to the framebuffer
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

/* Just like fragColor, the layout(location = x) annotations assign indices to the inputs that we can later use to reference
them. It is important to know that some types, like dvec3 64 bit vectors, use multiple slots. That means that the index after
it must be at least 2 higher (e.g. layout(location = 0) in dvec3 inPosition; layout(location = 2) in vec3 inColor; */

layout(location = 0) in vec2 inPosition;

//layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
//layout(location = 2) in vec2 inTexCoord;

/*layout(binding = 0) uniform UniformBufferObject 
{
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;*/

/*vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);*/

void main() 
{
    /* The main function is invoked for every vertex. The built-in gl_VertexIndex variable contains the index of the current
    vertex. This is usually an index into the vertex buffer, but in our case it will be an index into a hardcoded array of
    vertex data. The position of each vertex is accessed from the constant array in the shader and combined with dummy z and
    w components to produce a position in clip coordinates. The built-in variable gl_Position functions as the output */

    //gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    //fragColor = colors[gl_VertexIndex];

    //gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);

    //gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    //fragColor = inColor;
    //fragTexCoord = inTexCoord; // Sample colors from the texture

    // Compute shader (for vertex shader)
    gl_PointSize = 14.0;
    gl_Position = vec4(inPosition.xy, 1.0, 1.0);
    fragColor = inColor.rgb;
}