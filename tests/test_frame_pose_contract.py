"""CPU file/identity and analytic-camera controls; no human/model inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle, SCHEMA


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value): Path(p).write_text(json.dumps(value), encoding='utf-8')


class FramePoseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.K = np.array([[70., 2.7, 23.1], [0., 83., 17.3], [0., 0., 1.]])
        angle = .31
        self.E = np.eye(4)
        self.E[:3,:3] = [[np.cos(angle), 0, np.sin(angle)], [0,1,0], [-np.sin(angle),0,np.cos(angle)]]
        self.E[:3,3] = [.2, -.4, .7]
        self.rows=[]
        for i in range(3):
            image=self.root/f'image{i}.png'; mask=self.root/f'mask{i}.png'
            Image.new('RGB',(65,49),(50+i*20,70,110)).save(image)
            Image.new('L',(65,49),255).save(mask)
            self.rows.append(dict(frame=f'clip:{i*5}',source_frame=i*5,sequence_id='clip',video_sha256='a'*64,
                wh=[65,49],image=image.name,source_sha256=sha(image),mask=mask.name,mask_sha256=sha(mask),
                K_integer_centers=(self.K+np.diag([i,i,0])).tolist(),w2c=self.E.tolist()))
        order=[2,0,1]
        self.pose=dict(frame_ids=np.array([self.rows[i]['frame'] for i in order]),
            source_frames=np.array([self.rows[i]['source_frame'] for i in order]),
            image_sha256=np.array([self.rows[i]['source_sha256'] for i in order]),
            sequence_id=np.array('clip'),video_sha256=np.array('a'*64),
            vertices_world=np.array([np.array([[0,0,3],[1,0,3],[0,1,3]])+[i,0,0] for i in order],dtype=float),
            faces=np.array([[0,1,2]]),shared_betas=np.zeros((1,10)),
            K_integer_centers=np.array([self.rows[i]['K_integer_centers'] for i in order]),
            w2c=np.array([self.rows[i]['w2c'] for i in order]))
        self.manifest=dict(schema=SCHEMA,purpose='synthetic_contract_test',sequence_id='clip',video_sha256='a'*64,
            coordinate_system='opencv_world_meters',pixel_domain='rectified_pinhole',pixel_convention='integer_centers',
            reference_frame_id='clip:5',frames=dict(path='frames.json',rows_key='records'),poses=dict(path='poses.npz'))
        self.commit()

    def tearDown(self): self.tmp.cleanup()

    def commit(self):
        save(self.root/'frames.json',dict(records=self.rows))
        np.savez(self.root/'poses.npz',**self.pose)
        self.manifest['frames']['sha256']=sha(self.root/'frames.json')
        self.manifest['poses']['sha256']=sha(self.root/'poses.npz')
        save(self.root/'manifest.json',self.manifest)

    def load(self): return load_frame_pose_bundle(self.root/'manifest.json')
    def rejected(self):
        self.commit()
        with self.assertRaises(ValueError): self.load()

    def test_named_join_survives_independent_permutation(self):
        a=self.load(); self.rows.reverse();self.commit();b=self.load()
        for i in range(3):
            x,y=a.frame(f'clip:{i*5}'),b.frame(f'clip:{i*5}')
            self.assertEqual(x.pose_index,[1,2,0][i]);self.assertEqual(x.source_frame,i*5)
            np.testing.assert_array_equal(x.vertices_world,y.vertices_world)
            np.testing.assert_array_equal(x.K_integer_centers,self.rows[2-i]['K_integer_centers'])
        self.assertEqual(b.reference.frame_id,'clip:5')
        np.testing.assert_array_equal(b.reference.vertices_world[0],[1,0,3])

    def test_missing_pose_fails(self):
        self.pose['frame_ids'][0]='unknown';self.rejected()

    def test_duplicate_pose_fails(self):
        self.pose['frame_ids'][0]=self.pose['frame_ids'][1];self.rejected()

    def test_duplicate_roster_fails(self):
        self.rows[1]=copy.deepcopy(self.rows[0]);self.rejected()

    def test_cross_video_fails(self):
        self.pose['video_sha256']=np.array('b'*64);self.rejected()

    def test_wrong_sequence_fails(self):
        self.rows[0]['sequence_id']='other';self.rejected()

    def test_source_frame_mismatch_fails(self):
        self.pose['source_frames'][1]=777;self.rejected()

    def test_same_frame_wrong_camera_fails(self):
        self.pose['K_integer_centers'][1,0,2]+=.5;self.rejected()

    def test_pose_image_identity_mismatch_fails(self):
        self.pose['image_sha256'][1]='d'*64;self.rejected()

    def test_changed_image_fails_even_with_bound_manifest(self):
        Image.new('RGB',(65,49),'red').save(self.root/'image0.png')
        with self.assertRaises(ValueError): self.load()

    def test_changed_mask_fails(self):
        Image.new('L',(65,49),0).save(self.root/'mask0.png')
        with self.assertRaises(ValueError): self.load()

    def test_truncated_image_fails_with_updated_hash_chain(self):
        path=self.root/'image0.png'
        path.write_bytes(path.read_bytes()[:45])
        with Image.open(path) as opened:
            self.assertEqual(opened.size,(65,49))
        self.rows[0]['source_sha256']=sha(path)
        pose_index=self.pose['frame_ids'].tolist().index(self.rows[0]['frame'])
        self.pose['image_sha256'][pose_index]=sha(path)
        self.commit()
        with self.assertRaisesRegex(ValueError,'Cannot decode image input'):
            self.load()

    def test_truncated_mask_fails_with_updated_hash_chain(self):
        path=self.root/'mask0.png'
        path.write_bytes(path.read_bytes()[:45])
        with Image.open(path) as opened:
            self.assertEqual(opened.size,(65,49))
        self.rows[0]['mask_sha256']=sha(path)
        self.commit()
        with self.assertRaisesRegex(ValueError,'Cannot decode mask input'):
            self.load()

    def test_actual_dimensions_fail(self):
        self.rows[0]['wh']=[66,49];self.rejected()

    def test_reference_must_be_named_and_present(self):
        for value in ('absent','',0):
            with self.subTest(value=value):
                self.manifest['reference_frame_id']=value;self.rejected()

    def test_per_frame_identity_rejected(self):
        self.pose['shared_betas']=np.zeros((3,10));self.rejected()

    def test_extra_frame_beta_rejected(self):
        self.rows[0]['betas']=[0]*10;self.rejected()

    def test_optional_legacy_K_consistency(self):
        for row in self.rows:
            edge=np.array(row['K_integer_centers']);edge[:2,2]+=.5;row['K']=edge.tolist()
        self.commit();self.load()
        self.rows[0]['K'][0][2]-=.5;self.rejected()

    def test_empty_npz_is_explicit_error(self):
        (self.root/'poses.npz').write_bytes(b'')
        self.manifest['poses']['sha256']=sha(self.root/'poses.npz')
        save(self.root/'manifest.json',self.manifest)
        with self.assertRaises(ValueError): self.load()

    def test_invalid_camera_rejected(self):
        original=copy.deepcopy(self.rows[0])
        cases=[('focal',lambda r:r['K_integer_centers'][0].__setitem__(0,-3)),
               ('shear',lambda r:r['w2c'][0].__setitem__(1,.2)),
               ('nan',lambda r:r['K_integer_centers'][0].__setitem__(2,float('nan'))),
               ('bottomK',lambda r:r['K_integer_centers'][2].__setitem__(2,1.000001)),
               ('lowerK',lambda r:r['K_integer_centers'][1].__setitem__(0,.1))]
        for name,edit in cases:
            with self.subTest(name=name):
                self.rows[0]=copy.deepcopy(original);edit(self.rows[0]);self.rejected()

    def test_unknown_pixel_domain_not_guessed(self):
        self.manifest['pixel_domain']='fisheye';self.rejected()

    def test_wrong_pixel_convention_not_guessed(self):
        self.manifest['pixel_convention']='corners';self.rejected()

    def test_degenerate_topology_rejected(self):
        self.pose['faces'][0]=[0,0,2];self.rejected()

    def test_analytic_fullK_resize_and_renderer_pixel_mapping(self):
        f=self.load().frame('clip:0');c=f.for_size([31,27])
        world=np.array([[.2,.3,2.1],[-.4,.7,3.5],[1.2,-.3,5.]])
        camera=world@self.E[:3,:3].T+self.E[:3,3]
        native=camera@self.K.T;native=native[:,:2]/native[:,2,None]
        expected=(native+.5)*[31/65,27/49]-.5
        Ki=np.asarray(c['K_integer_centers']);Ke=np.asarray(c['K_renderer_corners'])
        uv=camera@Ki.T;uv=uv[:,:2]/uv[:,2,None]
        edge=camera@Ke.T;edge=edge[:,:2]/edge[:,2,None]
        ndc=2*edge/[31,27]-1;actual=((ndc+1)*[31,27]-1)/2
        np.testing.assert_allclose(uv,expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        self.assertNotEqual(Ki[0,1],0)
        C=np.eye(3);C[:2,2]=.5
        np.testing.assert_allclose(np.diag([31/65,27/49,1])@C@self.K,Ke,rtol=0,atol=1e-12)
        # Missing or duplicate +0.5 generates exactly half a pixel error.
        np.testing.assert_allclose(edge-uv,.5,rtol=0,atol=1e-12)
        np.testing.assert_allclose((camera-self.E[:3,3])@self.E[:3,:3],world,rtol=0,atol=1e-12)

    def test_no_alias_or_mutation(self):
        b=self.load();f=b.frame('clip:0')
        with self.assertRaises(ValueError): f.K_integer_centers[0,0]=0
        with self.assertRaises(ValueError): f.vertices_world.setflags(write=True)
        result=f.for_size([31,27])
        with self.assertRaises(ValueError): result['K_integer_centers'][0][0]=0
        result['K_integer_centers']=np.zeros((3,3))
        self.assertEqual(f.K_integer_centers[0,0],70)
        self.assertEqual(b.frame('clip:5').K_integer_centers[0,0],71)

    def test_invalid_output_size(self):
        f=self.load().frame('clip:0')
        for size in ([0,10],[3.4,8],[float('nan'),3]):
            with self.subTest(size=size),self.assertRaises(ValueError): f.for_size(size)

    def test_later_input_change_detected(self):
        b=self.load();(self.root/'frames.json').write_text('{}')
        with self.assertRaises(ValueError): b.assert_unchanged()

    def test_contract_never_admits_training(self):
        report=self.load().report()
        self.assertIs(report['physical_calibration_validated'],False)
        self.assertIs(report['training_admitted'],False)

    def test_cli_uses_reader_and_reports_only_contract(self):
        command=[sys.executable,'-B',str(ROOT/'scripts/check_frame_pose_inputs.py'),str(self.root/'manifest.json')]
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['training_admitted'])
        self.manifest['reference_frame_id']='missing';self.commit()
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,2)
        self.assertFalse(json.loads(result.stderr)['contract_passed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report');args=parser.parse_args()
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FramePoseTests))
    report=dict(kind='CPU frame/pose contract and analytic-camera checks',tests_run=result.testsRun,
                passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
                seconds=time.monotonic()-start,model_forwards=0,gaussian_steps=0,physical_quality_accepted=False,
                source_sha256=sha(ROOT/'src/shuolee_portrait/frame_pose_contract.py'),test_sha256=sha(__file__))
    if args.report: save(args.report,report)
    print(json.dumps(report));sys.exit(not result.wasSuccessful())
