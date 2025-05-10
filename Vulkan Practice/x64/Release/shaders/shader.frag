#version 450

/* The input variable does not necessarily have to use the same name, they will be linked together using the indexes specified
by the location directives */

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() 
{
    /* The main function is called for every fragment just like the vertex shader main function is called for every vertex.
    Colors in GLSL are 4-component vectors with the R, G, B and alpha channels within the [0, 1] range. Unlike gl_Position in
    the vertex shader, there is no built-in variable to output a color for the current fragment. You have to specify your own
    output variable for each framebuffer where the layout(location = 0) modifier specifies the index of the framebuffer. */

    outColor = vec4(fragColor, 1.0);
}