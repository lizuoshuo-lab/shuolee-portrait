"""New bounded geometry controls. No renderer, person model or training."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shuolee_portrait.framewise_clothed_surface import (
    NativePixelSquareBoundary, ReferenceGrid, VoteVolume, _boundary_margins,
    _quorum_margin, _required_votes, _move_queries, _mask_observation,
    _refine_native_edge_vertices)
from shuolee_portrait.free_clothing_transport import FreeClothingTransport


def frame(mask, valid=None, K=None):
    return dict(mask=mask, valid=np.ones_like(mask) if valid is None else valid,
                K=np.eye(3) if K is None else K, E=np.eye(4), wh=(mask.shape[1], mask.shape[0]))


def reference_motion():
    vertices=np.array([[[-.2,-.3,1.],[.4,-.1,1.],[.1,.4,1.]]])
    return FreeClothingTransport(frame_ids=('reference',), reference_frame_id='reference',
        vertices_world=vertices, faces=np.array([[0,1,2]]), motion_support_triangle_ids=np.array([0]),
        gaussian_ids=('sentinel',), frame_source_sha256={'reference':'1'*64}, translation_unit='meters')


class PixelBoundaryControls(unittest.TestCase):
    def test_single_pixel_corner_exact_square_not_interpolated_edt(self):
        mask=np.zeros((5,5),bool);mask[2,2]=True
        b=NativePixelSquareBoundary(mask,np.ones_like(mask))
        uv=np.array([[2.49,2.49],[2.,2.],[2.51,2.49],[2.51,2.51]])
        m,k=_boundary_margins(np.column_stack((uv,np.ones(len(uv)))),frame(mask),b)
        np.testing.assert_allclose(m,[.01,.5,-.01,-np.sqrt(.0002)],atol=1e-14)
        self.assertTrue(k.all())

    def test_segment_search_matches_independent_clamped_endpoint_projection(self):
        mask=np.zeros((17,19),bool);mask[3:14,2:16]=True;mask[6:10,7:11]=False
        boundary=NativePixelSquareBoundary(mask,np.ones_like(mask))
        rng=np.random.default_rng(194);uv=rng.uniform([-.49,-.49],[18.49,16.49],size=(257,2))
        # Brute-force vector projection on each finite segment, not KD logic.
        direction=np.eye(2)[boundary.extended_axis]
        a=boundary.centres-direction*.5;b=boundary.centres+direction*.5
        v=b-a;relative=uv[:,None]-a[None]
        t=np.clip((relative*v[None]).sum(-1)/(v*v).sum(-1),0,1)
        oracle=np.linalg.norm(relative-t[:,:,None]*v[None],axis=-1).min(-1)
        np.testing.assert_allclose(boundary.distance(uv),oracle,atol=1e-14)

    def test_unknown_pixels_and_image_exterior_are_not_boundary_walls(self):
        mask=np.zeros((5,5),bool);mask[2,2]=True
        valid=np.zeros_like(mask);valid[2,2]=True
        boundary=NativePixelSquareBoundary(mask,valid)
        self.assertEqual(len(boundary.centres),0)
        margins,known=_boundary_margins(np.array([[2.,2.,1.],[1.,2.,1.],[-1.,2.,1.]]),frame(mask,valid),boundary)
        self.assertTrue(np.isposinf(margins[0]));self.assertFalse(known[1:].any())
        self.assertTrue(np.isnan(margins[1:]).all())
        # Two classes separated solely by unknown are explicitly unresolved.
        valid[0,0]=True
        with self.assertRaisesRegex(ValueError,'only by unknown'):
            NativePixelSquareBoundary(mask,valid).distance(np.array([[2.,2.]]))

    def test_quorum_preserves_strict_float_division_with_unknowns(self):
        for tau in (.58,.9,.6):
            required=_required_votes(50,tau)
            for n in range(1,51):
                expected=next(p for p in range(n+1) if p/n > tau)
                self.assertEqual(required[n],expected)
        margins=np.array([[1.,1.],[2.,2.],[-3.,np.nan],[-4.,np.nan]])
        known=np.isfinite(margins)
        values=_quorum_margin(margins,known,min_valid_views=2,vote_fraction=.6)
        np.testing.assert_allclose(values,[-3.,1.])

    def test_complete_skew_K_and_nontrivial_rigid_camera(self):
        mask=np.zeros((21,23),bool);mask[4:17,5:18]=True
        f=frame(mask,K=np.array([[10.,2.,11.2],[0.,11.,9.8],[0.,0.,1.]]))
        angle=.37;c,s=np.cos(angle),np.sin(angle)
        f['E'][:3,:3]=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        f['E'][:3,3]=[.2,-.1,.3]
        uv=np.array([[6.1,5.2],[17.9,6.],[2.1,7.]])
        camera=(np.column_stack((uv,np.ones(3)))@np.linalg.inv(f['K']).T)*np.array([2.,3.,4.])[:,None]
        world=(camera-f['E'][:3,3])@f['E'][:3,:3]
        m,k=_boundary_margins(world,f,NativePixelSquareBoundary(mask,f['valid']))
        hit,known=_mask_observation(world,f)
        self.assertTrue(np.array_equal(m>0,hit));self.assertTrue(np.array_equal(k,known))
        np.testing.assert_allclose(m,[1.6,-.4,-2.4],atol=1e-12)

    def test_distinct_motion_supports_are_not_blended(self):
        v=np.array([[-2.3,-.3,5.],[-1.6,-.1,5.],[-1.9,.4,5.],
                    [1.7,-.3,5.],[2.4,-.1,5.],[2.1,.4,5.]])
        q0=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        q1=np.diag([-1.,-1.,1.])
        posed=v.copy()
        for ids,Q,shift in ((slice(0,3),q0,[.3,.2,.1]),(slice(3,6),q1,[-.2,.1,-.1])):
            centre=v[ids].mean(0);posed[ids]=(v[ids]-centre)@Q.T+centre+shift
        motion=FreeClothingTransport(frame_ids=('move','reference'),reference_frame_id='reference',
            vertices_world=np.stack((posed,v)),faces=np.array([[0,1,2],[3,4,5]]),
            motion_support_triangle_ids=np.array([0,1]),gaussian_ids=('a','b'),
            frame_source_sha256={'move':'1'*64,'reference':'2'*64},translation_unit='meters')
        xyz=np.array([[-1.8,.1,5.3],[2.2,.1,5.3]])
        expected=np.stack(((xyz[0]-v[:3].mean(0))@q0.T+posed[:3].mean(0),
                           (xyz[1]-v[3:].mean(0))@q1.T+posed[3:].mean(0)))
        np.testing.assert_allclose(_move_queries(motion,'move',xyz,np.array([0,1])),expected,atol=1e-14)

    def test_finite_scan_reports_multiple_crossings_and_unknown_intervals(self):
        motion=reference_motion();tree=cKDTree(motion._centroids[0])
        mask=np.tile(np.arange(65)%2==0,(65,1));valid=np.ones_like(mask);valid[:,38]=False
        f=frame(mask,valid,K=np.array([[16.,0.,32.],[0.,16.,32.],[0.,0.,1.]]))
        grid=ReferenceGrid(([-1.,0,1.],[-1.,0,1.],[0.,1.,2.]),reference_frame_id='reference')
        volume=VoteVolume(grid,np.ones(grid.shape,np.uint32),np.ones(grid.shape,np.uint32),
                          np.zeros(grid.size,np.int64),1,.9,'{}')
        xyz,audit=_refine_native_edge_vertices(np.array([[.35,0.,1.]]),volume,motion,tree,{'reference':f})
        self.assertEqual(audit['multiple_observed_crossing_edges'],1)
        self.assertEqual(audit['scan_edges_with_unknown'],1)
        self.assertEqual(audit['refined_vertices'],1)
        self.assertTrue(_mask_observation(xyz,f)[0][0])
        self.assertFalse(audit['subscan_crossings_certified_absent'])


if __name__=='__main__':
    unittest.main(verbosity=2)
