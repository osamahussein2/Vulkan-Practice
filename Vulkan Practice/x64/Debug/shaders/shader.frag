#version 450

/* The input variable does not necessarily have to use the same name, they will be linked together using the indexes specified
by the location directives */

layout(location = 0) in vec3 fragColor;
//layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

//layout(binding = 1) uniform sampler2D texSampler;

void main() 
{
    /* The main function is called for every fragment just like the vertex shader main function is called for every vertex.
    Colors in GLSL are 4-component vectors with the R, G, B and alpha channels within the [0, 1] range. Unlike gl_Position in
    the vertex shader, there is no built-in variable to output a color for the current fragment. You have to specify your own
    output variable for each framebuffer where the layout(location = 0) modifier specifies the index of the framebuffer. */

    //outColor = vec4(fragColor, 1.0);

    // The fragTexCoord values will be smoothly interpolated across the area of the square by the rasterizer
    //outColor = vec4(fragTexCoord, 0.0, 1.0);

    //outColor = texture(texSampler, fragTexCoord);

    // Manipulate the texture colors using vertex colors
    //outColor = vec4(fragColor * texture(texSampler, fragTexCoord).rgb, 1.0);

    // Compute shader (for fragment shader)
    vec2 coord = gl_PointCoord - vec2(0.5);
    outColor = vec4(fragColor, 0.5 - length(coord));

}