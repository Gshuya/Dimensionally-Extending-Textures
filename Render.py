import mitsuba as mi

# Set the variant of the renderer
mi.set_variant('scalar_rgb')
# Load a scene
scene = mi.load_file('scenes/matpreview/scene.xml')
# Render the scene
img = mi.render(scene)
mi.util.write_bitmap("3DRender/4.png", img)