"""Native36-view review from the immutable18k checkpoint, independent of PLY."""
import inspect
import diagnose_indoor_circle16_export124 as prior

def main():
    code=inspect.getsource(prior.main)
    code=code[:code.index('        portable=render(cam,plain,pipe,bg)')]
    code=code.replace("out=ROOT/'export-failure-diagnostic'","out=ROOT/'native-checkpoint-orbit18k'")
    code+='''        from PIL import Image
        R=np.diag([1.,-1.,-1.])@np.asarray(rows[0]['w2c'])[:3,:3]
        verts=model.shared_offset_vertices(0).cpu().numpy()@R.T
        center=(verts.min(0)+verts.max(0))*.5
        radius=max(float(np.ptp(verts,axis=0).max())*1.8,.1)
        sourceE=np.asarray(rows[0]['w2c'])
        eye0=(-sourceE[:3,:3].T@sourceE[:3,3])@R.T
        direction=(eye0-center)/np.linalg.norm(eye0-center)
        transform=np.eye(4);transform[:3,:3]=R
        records=[]
        for i,angle in enumerate(np.linspace(0,2*np.pi,36,endpoint=False)):
            yaw=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
            eye=center+radius*(yaw@direction)
            forward=center-eye;forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
            down=np.cross(forward,right)
            E=np.eye(4);E[:3,:3]=np.stack((right,down,forward));E[:3,3]=-E[:3,:3]@eye
            row=dict(frame=f'novel_{i:03d}',wh=[384,512],K=[[450.,0,192.],[0,450.,256.],[0,0,1]],w2c=(E@transform).tolist(),synthetic=True)
            camera_i,_=camera(row,i,pose_index=0)
            image(render(camera_i,model,pipe,bg)['render']).save(out/(row['frame']+'.png'))
            records.append(row)
        pages=[]
        for start in (0,12,24):
            canvas=Image.new('RGB',(1536,1536),'black')
            for j in range(12):
                canvas.paste(Image.open(out/f'novel_{start+j:03d}.png'),((j%4)*384,(j//4)*512))
            path=out/f'orbit_{start:02d}-{start+11:02d}.png';canvas.save(path)
            pages.append(dict(path=str(path),sha256=sha(path)))
        save(out/'result.json',dict(updated=now(),checkpoint_sha256=sha(snap/'checkpoint.pt'),
            cameras=records,pages=pages,reference_frame=rows[0]['frame'],
            views=36,actual_visual_review=False,quality_accepted=False,
            renderer='Original bound-Gaussian native render from exact checkpoint tensors; no PLY conversion',
            ground_truth_available=False,optimizer_updates_added=0))
        for path,digest in hashes.items():assert sha(path)==digest
        print('NATIVE_CHECKPOINT_36_VIEWS_READY',flush=True)
'''
    ns=dict(prior.main.__globals__)
    exec(compile(code,__file__+':immutable-native-replay','exec'),ns)
    ns['main']()

if __name__=='__main__':main()
