"""Render one recorded residual camera without overriding K, E, or coefficients."""
import argparse
from native_scene_package122_v5 import load_scene,render_recorded,save_render

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True);p.add_argument('--camera',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();scene=load_scene(a.scene)
    rgb,alpha,receipt=render_recorded(scene,a.camera)
    receipt.update(projection=scene.manifest['camera_contract']['projection'],
        canonical_parameters='native-parameters.npz',camera=scene.camera(a.camera))
    print(save_render(a.output,rgb,alpha,receipt))

if __name__=='__main__':main()
