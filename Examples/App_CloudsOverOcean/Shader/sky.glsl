uniform mat4 screenToCamera;
uniform mat4 cameraToWorld;
uniform vec3 worldCamera;
uniform vec3 worldSunDir;

varying vec3 viewDir;

#ifdef _VERTEX_

void main() {
    viewDir = (cameraToWorld * vec4((screenToCamera * gl_Vertex).xyz, 0.0)).xyz;
    gl_Position = vec4(gl_Vertex.xy, 0.9999999, 1.0);
}

#endif

#ifdef _FRAGMENT_

void main() {
    vec3 v = normalize(viewDir);
    vec3 sunColor = vec3(step(cos(3.1415926 / 180.0), dot(v, worldSunDir))) * SUN_INTENSITY;
    vec3 extinction;
    vec3 inscatter = skyRadiance(worldCamera + earthPos, v, worldSunDir, extinction);
    vec3 finalColor = sunColor * extinction + inscatter;
    gl_FragColor.rgb = hdr(finalColor);
    gl_FragColor.a = 1.0;
}

#endif
