"""PTS identity and real image/camera-domain controls; no model inference."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from shuolee_portrait import preprocessing as p, runtime as rt
from shuolee_portrait.video119 import prepare_video


class PixelContractTests(unittest.TestCase):
    def test_all_four_rules_project_the_same_3D_points(self):
        expected = { (1920,1080): ([960,540],[960,540]),
                     (3840,2160): ([960,540],[1920,1080]),
                     (2704,1520): ([1352,760],[2704,1520]),
                     (1520,2704): ([760,1352],[1520,2704]) }
        xyz = np.array([[.2,-.3,2.], [.7,.4,3.], [-.5,.1,1.4]])
        for wh, (pose_wh, gs_wh) in expected.items():
            with self.subTest(wh=wh):
                c = p.resolution_contract(list(wh))
                self.assertEqual((c['pose_wh'],c['gaussian_wh']), (pose_wh,gs_wh))
                K = np.array([[1100.,0,wh[0]/2+3.27],[0,1090.,wh[1]/2-2.71],[0,0,1.]])
                pose_K = np.asarray(c['source_to_pose_integer']) @ K
                result = p.camera_intrinsics(c, pose_K)
                expected_K = np.asarray(c['source_to_GS_integer']) @ K
                np.testing.assert_allclose(result['GS_K_integer_centers'], expected_K, atol=1e-10)
                a, b = xyz @ pose_K.T, xyz @ expected_K.T
                pose_uv, gs_uv = a[:,:2]/a[:,2:], b[:,:2]/b[:,2:]
                actual = p.map_pixels(pose_uv, c['pose_to_GS_integer'])
                np.testing.assert_allclose(actual, gs_uv, atol=1e-10)
                scale = gs_wh[0]/pose_wh[0]
                np.testing.assert_allclose(gs_uv, (pose_uv+.5)*scale-.5, atol=1e-10)
                np.testing.assert_allclose(np.diag([scale,scale,1.]) @ result['pose_K_corner'],
                                           result['GS_K_corner'], atol=1e-10)

    def test_unknown_rule_or_bad_intrinsic_bridge_is_rejected(self):
        for wh in ([1280,720], [2704.,1520], [0,1520], [True,1520]):
            with self.subTest(wh=wh), self.assertRaises(ValueError):
                p.resolution_contract(wh)
        c = p.resolution_contract([2704,1520])
        c['pose_to_GS_integer'][0][2] = 0  # Missing principal-point translation.
        with self.assertRaisesRegex(ValueError, 'Integer/corner'):
            p.camera_intrinsics(c, np.eye(3))

    def test_variable_pts_and_nonzero_start_are_used_instead_of_stride(self):
        times = np.r_[np.arange(8)/24, 8/24+np.arange(12)/15] + 12.5
        targets, ids = p.sampling_indices(times)
        expected = [min(range(len(times)), key=lambda i: (abs(times[i]-t), i)) for t in targets]
        self.assertEqual(ids.tolist(), expected)
        self.assertEqual(targets[0], times[0])
        self.assertFalse(np.all(np.diff(ids) == 8))
        self.assertEqual(len(set(ids)), len(ids))

    def test_earlier_frame_wins_tie_and_invalid_pts_are_not_fabricated(self):
        grids, ids = p.sampling_indices([0., .25, 5/12, .75])
        self.assertEqual(ids[1], 1)
        for times in ([0.], [0.,0.], [1.,0.], [0.,float('nan')], [0.,1.,2.]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                p.sampling_indices(times)
        for fps in (True, 0, 2, 3.1):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                p.sampling_indices([0.,.1,.2], fps)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class RealDecodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def clip(self, wh):
        path = self.root/f'{wh[0]}x{wh[1]}.mkv'
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-v','error','-n','-f','lavfi',
            '-i',f'testsrc2=size={wh[0]}x{wh[1]}:rate=6:duration=1',
            '-c:v','ffv1','-threads','2',str(path)], check=True)
        return path

    def workspace(self, name):
        path = self.root/name; (path/'logs').mkdir(parents=True); (path/'data').mkdir()
        return path

    def test_actual_decode_all_rules_preserves_source_indices_and_pixels(self):
        for wh in p.RESOLUTION_RULES:
            with self.subTest(wh=wh):
                video = self.clip(wh); root = self.workspace(str(wh))
                c = p.extract_dual_resolution(video,root,dict(sample_fps=3,min_frames=3,max_frames=300))
                gs, pose = rt.read(root/'decoded.json'), rt.read(root/'decoded-pose.json')
                self.assertEqual(gs['source_indices'], [0,2,4])
                self.assertEqual(gs['source_indices'], pose['source_indices'])
                raw = subprocess.check_output(['ffmpeg','-nostdin','-hide_banner','-v','error',
                    '-noautorotate','-i',str(video),'-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
                source = np.frombuffer(raw,np.uint8).reshape(wh[1],wh[0],3)
                for manifest,size in ((gs,c['gaussian_wh']),(pose,c['pose_wh'])):
                    row=manifest['records'][0]
                    self.assertEqual(row['wh'], size)
                    self.assertEqual(rt.sha(row['image']),row['source_sha256'])
                    with Image.open(row['image']) as im:
                        np.testing.assert_array_equal(np.asarray(im),cv2.resize(source,size,interpolation=cv2.INTER_AREA))
                with self.assertRaises(FileExistsError):
                    p.extract_dual_resolution(video,root,dict(sample_fps=3,min_frames=3,max_frames=300))

    def test_camera_frontend_uses_pose_images_and_maps_full_K_to_GS(self):
        video=self.clip((2704,1520));root=self.workspace('pipeline')
        K=np.array([[1100.,0,1370.25],[0,1090.,740.75],[0,0,1.]])
        profile=dict(mode='fixed_pinhole',image_wh=[2704,1520],K=K.tolist(),
                     provenance='Synthetic test camera',pixel_convention='integer_centers')
        cfg=dict(sample_fps=3,min_frames=3,max_frames=300,intrinsics=profile)
        def tracks(_root,rows,_out,_status,_masks):
            self.assertTrue(all(r['wh']==[1352,760] and Path(r['image']).parent.name=='pose' for r in rows))
            return [],[]
        def solve(_xy,_tracks,Ki,*args):
            np.testing.assert_allclose(Ki, [[550,0,684.875],[0,545,370.125],[0,0,1]])
            E=np.repeat(np.eye(4)[None],3,axis=0);E[:,0,3]=[0,.1,.2]
            return E,dict(retained_input_indices=[0,1,2])
        with patch('shuolee_portrait.video119._fresh_tracks',side_effect=tracks), \
             patch('shuolee_portrait.video119.solve_camera_geometry',side_effect=solve):
            report=prepare_video(video,root,cfg,None,camera_only=True)
        self.assertFalse(report['observations_complete'])
        gs=rt.read(root/'data/inputs.json');pose=rt.read(root/'data/pose-inputs.json')
        for a,b in zip(gs['records'],pose['records']):
            np.testing.assert_array_equal(a['K_integer_centers'],K)
            self.assertEqual(a['w2c'],b['w2c'])
            self.assertEqual(a['source_frame_index'],b['source_frame_index'])
        # Synthetic native detections exercise the actual coordinate adapter;
        # no detector, fitting model or Gaussian optimizer is loaded.
        masks=root/'data/masks';masks.mkdir()
        for r in gs['records']:
            mask=masks/r['name'];Image.fromarray(np.full((1520,2704),255,np.uint8)).save(mask)
            r.update(mask=str(mask),mask_sha256=rt.sha(mask))
        rt.save(root/'data/inputs.json',gs)
        keypoints=np.tile([[123.25,456.75]],(3,133,1))+np.arange(3)[:,None,None]*10
        scores=np.repeat(np.array([.3,.6,.9])[:,None],133,axis=1)
        obs=dict(frame_ids=np.array([r['frame'] for r in reversed(gs['records'])]),
                 keypoints=keypoints[::-1],scores=scores[::-1],boxes=np.tile([10.,20.,100.,200.],(3,1)),
                 flip_disagreement=np.ones((3,133)),joint_names=np.array(['joint']*133))
        np.savez_compressed(root/'data/observations.npz',**obs)
        p.derive_pose_observations(root)
        with np.load(root/'data/pose-observations.npz',allow_pickle=False) as z:
            np.testing.assert_allclose((z['keypoints']+.5)*2-.5,keypoints)
            np.testing.assert_array_equal(z['scores'],scores)
            np.testing.assert_array_equal(z['flip_disagreement'],obs['flip_disagreement']/2)
            self.assertEqual(z['frame_ids'].tolist(),[r['frame'] for r in gs['records']])
        for r in rt.read(root/'data/pose-inputs.json')['records']:
            with Image.open(r['mask']) as im:self.assertEqual(im.size,(1352,760))
            self.assertEqual(r['native_image_sha256'],next(g['source_sha256'] for g in gs['records'] if g['frame']==r['frame']))


if __name__ == '__main__':
    unittest.main()
