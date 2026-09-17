# Проверка, что Cycles видит GPU (OPTIX/CUDA). Запуск: blender -b --factory-startup --python /opt/lct/blender-smoke.py
import bpy
p = bpy.context.preferences.addons["cycles"].preferences
found = []
for kind in ("OPTIX", "CUDA"):
    try:
        p.compute_device_type = kind
        p.get_devices()
        found += [(d.name, d.type) for d in p.devices if d.type == kind]
    except Exception as e:  # noqa: BLE001
        print("cycles", kind, "error", e)
print("SMOKE_BLENDER", "OK" if found else "FAIL", found)
