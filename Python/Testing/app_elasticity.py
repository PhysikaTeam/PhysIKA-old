import PyPhysIKA as pk

log = pk.Log()
log.set_output("output.txt")
inst = pk.SceneGraph.get_instance()
el = pk.ParticleElasticBody3f()

el.load_particles("../../Media/bunny/bunny_points.obj")
trans1 = pk.Vector3f([0.5, 0.2, 0.5])
el.translate(trans1)

bound1 = pk.StaticBoundary3f()
bound1.load_cube(pk.Vector3f([0, 0, 0]), pk.Vector3f([1, 1, 1]), 0.005, True, False)
inst.set_root_node(bound1)
bound1.add_particle_system(el)

r1 = pk.PointRenderModule()
el.add_visual_module(r1)

app = pk.GLApp()
app.create_window(800, 600)
app.main_loop()