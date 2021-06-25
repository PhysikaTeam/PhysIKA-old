uniform mat4 worldToScreen;
uniform vec3 worldCamera;
uniform vec3 worldSunDir;

varying vec3 worldP;

#ifdef _VERTEX_

void main() {
    gl_Position = worldToScreen * vec4(gl_Vertex.xyz, 1.0);
    worldP = gl_Vertex.xyz;
}

#endif

#ifdef _FRAGMENT_

void main() {
    gl_FragColor = cloudColor(worldP, worldCamera, worldSunDir);
    gl_FragColor.rgb = hdr(gl_FragColor.rgb);
}

#endif
