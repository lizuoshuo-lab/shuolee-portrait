"""Compare source-marked pinnae with the existing 022 fit, without fitting."""
from pathlib import Path
import pickle

import cv2
import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from prepare_leafwoman_ear139 import ARCHIVE, MAP, MASK, OUT, atomic, read
from review_full_sequence123 import joined


class EarMaskReader(pickle.Unpickler):
    TYPES = {('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
             ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}

    def find_class(self, module, name):
        return self.TYPES[(module,name)]


def crop_visible_faces(xy, z, faces, wh):
    """Label closest face at each native integer pixel center in a small ROI."""
    w,h = wh
    depth = np.zeros((h,w), np.float64)
    owner = np.full((h,w), -1, dtype=np.int32)
    corners = xy[faces]
    lo = np.maximum(np.ceil(corners.min(1)).astype(int), 0)
    hi = np.minimum(np.floor(corners.max(1)).astype(int), [w-1,h-1])
    keep = (hi >= lo).all(1)
    for i in np.flatnonzero(keep):
        a,b,c = corners[i]
        ab,ac = b-a,c-a
        det = ab[0]*ac[1]-ab[1]*ac[0]
        if abs(det) < 1e-10:
            continue
        x0,y0 = lo[i]
        x1,y1 = hi[i]+1
        yy,xx = np.ogrid[y0:y1,x0:x1]
        bx,by = xx-a[0],yy-a[1]
        u = (bx*ac[1]-by*ac[0])/det
        v = (ab[0]*by-ab[1]*bx)/det
        a_z,b_z,c_z = z[faces[i]]
        inverse_z = (1-u-v)/a_z+u/b_z+v/c_z
        old = depth[y0:y1,x0:x1]
        update = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (inverse_z > old)
        old[update] = inverse_z[update]
        owner[y0:y1,x0:x1][update] = i
    return owner


def raster_control():
    xy = np.array([[1,1],[8,1],[1,8]]*2, dtype=float)
    z = np.array([3.,3.,3.,2.,2.,2.])
    f = np.arange(6).reshape(2,3)
    a = crop_visible_faces(xy,z,f,(10,10))
    b = crop_visible_faces(xy,z,f[::-1,::-1],(10,10))
    assert a[3,3] == 1 and b[3,3] == 0 and a[8,8] == b[8,8] == -1
    assert np.array_equal(a == 1, b == 0)


def masks(plan):
    assert sha(MASK) == plan['ear_mask_sha256'] == 'ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492'
    assert sha(MAP) == plan['correspondence_sha256'] == '7e70cdc3659aae699b9732e8dd4af49106310c69b90dc83d9f73e96dbf871e49'
    with MASK.open('rb') as f:
        regions = EarMaskReader(f, encoding='latin1').load()
    idx = np.load(MAP, allow_pickle=False)
    assert idx.shape == (5023,) and idx.dtype.kind in 'iu' and len(np.unique(idx)) == len(idx)
    assert idx.min() >= 0 and idx.max() < 10475
    ears = {}
    for side, vertex in [('left',6),('right',616)]:
        region = regions[side+'_ear']
        assert region.dtype.kind in 'iu' and region.min() >= 0 and region.max() < len(idx)
        selected = idx[region]
        assert len(selected) == len(np.unique(selected)) == 588 and vertex in selected
        ears[side] = selected
    assert not np.intersect1d(ears['left'],ears['right']).size
    return ears


def annotate_image(raw, regions, mesh, target, vertex):
    pixels = np.array(raw)
    pixels[mesh] = (0.48*pixels[mesh] + 0.52*np.array([30,190,220])).astype(np.uint8)
    output = Image.fromarray(pixels)
    d = ImageDraw.Draw(output)
    for polygon,color in zip(regions, ('#ffe341','#ff9e43')):
        points = [tuple(row) for row in polygon.tolist()]
        d.line(points+[points[0]], fill=color, width=1)
    for xy,color in ((target,'#ff37ba'),(vertex,'#10ecff')):
        x,y = map(float,xy)
        d.ellipse((x-3,y-3,x+3,y+3), outline='#202020', width=2)
        d.ellipse((x-3,y-3,x+3,y+3), outline=color, width=1)
    return output


def main():
    cv2.setNumThreads(1)
    assert not (OUT/'result.json').exists()
    raster_control()
    p = OUT/'plan.json'
    plan = read(p)
    assert sha(ARCHIVE) == plan['archive_sha256']
    annotations = [OUT/('source-only-annotation-'+c+'.json') for c in 'ab']
    readings = [read(q) for q in annotations]
    assert all(r['plan_sha256'] == sha(p) for r in readings)
    assert all([a['frame'] for a in r['rows']] == [a['frame'] for a in plan['frames']] for r in readings)
    ear_map = masks(plan)
    rows = []
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        ids = archive['frame_ids'].tolist()
        faces, vertices = archive['faces'], archive['vertices_camera']
        joints, targets = archive['joints_projected_native'], archive['observed_native_keypoints']
        K = archive['K_integer_centers']
        for item, a, b in zip(plan['frames'], *(r['rows'] for r in readings), strict=True):
            frame,side = item['frame'],item['side']
            assert a['side'] == b['side'] == side
            path = Path(item['original_crop'])
            assert sha(path) == item['crop_sha256']
            with Image.open(path) as im:
                raw = im.convert('RGB')
            origin = np.array(item['crop_xyxy'][:2])
            i = ids.index(frame)
            j,v = (18,6) if side == 'left' else (17,616)
            xyz = vertices[i].astype(np.float64)
            z = xyz[:,2]
            assert np.isfinite(xyz).all() and z.min() > .05
            pxyz = xyz@K.T
            uv = pxyz[:,:2]/pxyz[:,2:]-origin
            assert np.max(np.abs(uv[v]+origin-joints[i,j])) < .001
            ear_faces = np.isin(faces,ear_map[side]).all(1)
            assert ear_faces.sum() > 100
            owner = crop_visible_faces(uv,z,faces,raw.size)
            mesh_mask = (owner >= 0) & ear_faces[np.maximum(owner,0)]
            assert mesh_mask.any()
            dist = cv2.distanceTransform((~mesh_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            target, vertex = targets[i,j,:2]-origin,uv[v]
            polygons, report, masks2d = [], [], []
            for label in (a,b):
                poly = np.asarray(label['polygon_xy'],dtype=np.int32)
                tol = label['uncertainty_px']
                assert poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 6 and 0 < tol <= 10
                assert (poly >= 0).all() and (poly < np.array(raw.size)).all()
                reference = np.zeros((raw.height,raw.width),dtype=np.uint8)
                cv2.fillPoly(reference,[poly],1)
                reference = reference.astype(bool)
                assert reference.any()
                signed = lambda xy: cv2.pointPolygonTest(poly,tuple(map(float,xy)),True)
                report.append(dict(source_pixels=int(reference.sum()), vertex_signed_distance_px=signed(vertex),
                    detector_signed_distance_px=signed(target), toleranced_source_region_coverage=float(np.mean(dist[reference] <= tol)),
                    source_region_to_mesh_P90_px=float(np.quantile(dist[reference],.9)),uncertainty_px=tol))
                polygons.append(poly)
                masks2d.append(reference)
            figure = OUT/(frame+'-ear-region.png')
            joined([raw, annotate_image(raw,polygons,mesh_mask,target,vertex)]).save(figure)
            rows.append(dict(frame=frame,side=side,readings=report,
                source_reading_IoU=float(np.sum(masks2d[0]&masks2d[1])/np.sum(masks2d[0]|masks2d[1])),
                mesh_visible_patch_pixels=int(mesh_mask.sum()), view_image=str(figure), image_sha256=sha(figure)))
    atomic(OUT/'result.json',dict(rows=rows,plan_sha256=sha(OUT/'plan.json'),rasterizer_synthetic_control_passed=True,
        annotation_sha256=[sha(q) for q in annotations], camera_refit=False, new_geometry=False, gaussian_started=False,
        native_projection_model='022 frozen conditional full K, native integer pixels', physical_truth=False,
        model_self_occlusion_only=True, mesh_hair_occlusion_modeled=False))
    for row in rows:
        print(row['frame'],row['side'],[(round(x['vertex_signed_distance_px'],1),round(x['toleranced_source_region_coverage'],3)) for x in row['readings']])


if __name__=='__main__':
    main()
