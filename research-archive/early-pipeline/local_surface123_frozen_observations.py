"""Read immutable native tracks without conflating split and physical role.

This is a transport/admission boundary, not a physical truth detector. Evidence
must be reviewed independently; successful parsing never admits human training.
The historical N3 schema and saved observations are deliberately unchanged.
"""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import hashlib,json,math
import numpy as np

SCHEMA='shuolee-frozen-camera-observations/1'

class ObservationContractError(RuntimeError):pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def checked_file(reference,base):
    if not isinstance(reference,dict) or not reference.get('path') or not reference.get('sha256'):
        raise ObservationContractError('missing_file_binding')
    p=Path(reference['path']);p=p if p.is_absolute() else base/p
    if not p.is_file() or sha(p)!=reference['sha256']:
        raise ObservationContractError('file_hash_mismatch: '+str(p))
    return p.resolve()

def fail(condition,message):
    if not condition:raise ObservationContractError(message)

def check_material_support(path,source_sha256,segment,frames,xy,role):
    """Evidence files must cover these exact nodes, not another video/track.

    This verifies the scope of claims only. Their physical truth still comes
    from the independent measurement/review represented by the evidence file.
    """
    e=read(path)
    fail(e.get('schema')=='shuolee-material-support/1','material_support_schema')
    fail(e.get('source_video_sha256')==source_sha256,'material_support_source_mismatch')
    fail(e.get('segment_id')==segment['id'] and e.get('family_id')==segment['family_id'],'material_support_track_mismatch')
    fail(e.get('material_role')==role,'material_support_role_mismatch')
    fail(e.get('material_identity')=='confirmed' and e.get('motion_state')=='bounded_static','material_support_claims_incomplete')
    fail(isinstance(e.get('basis'),str) and bool(e['basis'].strip()),'material_support_basis_missing')
    nodes={}
    for v in e.get('observations',[]):
        node=(v.get('slot'),v.get('keypoint'))
        fail(node not in nodes and list(node) in segment['nodes'],'material_support_node_mismatch')
        i,k=node;f=frames[i]
        fail(v.get('source_frame')==f['source_frame'],'material_support_frame_mismatch')
        fail(v.get('image_sha256')==f['image_sha256'] and v.get('feature_sha256')==f['feature_sha256'],'material_support_bytes_mismatch')
        fail(np.array_equal(np.asarray(v.get('uv'),dtype=np.float64),xy[i][k]),'material_support_pixel_mismatch')
        uncertainty=v.get('measurement_uncertainty_px')
        fail(v.get('visibility')=='visible','material_support_visibility_unknown')
        fail(isinstance(uncertainty,(int,float)) and not isinstance(uncertainty,bool) and math.isfinite(uncertainty) and uncertainty>=0,'material_support_uncertainty_unknown')
        nodes[node]=v
    fail(set(nodes)=={tuple(n) for n in segment['nodes']},'material_support_incomplete_coverage')

@dataclass(frozen=True)
class NativeObservation:
    slot:int
    source_frame:int
    keypoint_id:int
    source_image_sha256:str
    feature_file_sha256:str
    native_xy:tuple

@dataclass(frozen=True)
class FrozenTrack:
    segment_id:str
    family_id:str
    partition:str
    development_only:bool
    observations:tuple
    material_role:str
    family_conflict:bool
    fit_reasons:tuple

@dataclass(frozen=True)
class FrozenInput:
    manifest_sha256:str
    source_video:Path
    source_video_sha256:str
    subject_id:str
    purpose:str
    decoded:dict
    frames:tuple
    xy:tuple
    tracks:tuple
    bound_files:tuple

    @property
    def fit_tracks(self):return tuple(t for t in self.tracks if not t.fit_reasons)

    @property
    def held_tracks(self):return tuple(t for t in self.tracks if t.partition=='HELD')

    def require_fit_support(self):
        fail(bool(self.fit_tracks),'no_reviewed_static_background_FIT_tracks')
        return self.fit_tracks

    def audit(self):
        return dict(schema=SCHEMA,subject=self.subject_id,manifest_sha256=self.manifest_sha256,
            source_video_sha256=self.source_video_sha256,frames=len(self.frames),
            original_tracks=len(self.tracks),original_observations=sum(len(t.observations) for t in self.tracks),
            fit_tracks=len(self.fit_tracks),held_tracks=len(self.held_tracks),
            family_conflicts=len({t.family_id for t in self.tracks if t.family_conflict}),
            conflicting_segments=sum(t.family_conflict for t in self.tracks),
            rejected_reasons=dict(Counter(r for t in self.tracks for r in t.fit_reasons)),
            original_partitions_preserved=True,native_pixels_preserved=True,
            feature_extraction=False,matching=False,physical_accuracy_proven=False,
            human_fitting_admitted=False,training_admitted=False)

def load_subject_input(subject):
    """The public caller requires an independently pinned manifest digest."""
    ref=subject.get('frozen_observations')
    fail(isinstance(ref,dict),'frozen_observations_required_no_legacy_extraction_fallback')
    manifest=checked_file(ref,Path.cwd())
    return load_frozen_input(manifest,ref['sha256'],subject)

def load_frozen_input(manifest,expected_sha256,subject):
    manifest=Path(manifest).resolve();base=manifest.parent
    fail(sha(manifest)==expected_sha256,'manifest_hash_mismatch')
    doc=read(manifest)
    fail(doc.get('schema')==SCHEMA,'unsupported_observation_schema')
    fail(doc.get('subject_id')==subject['id'],'subject_identity_mismatch')
    fail(doc.get('pixel_domain')=='native_exported_MP4_integer_centers','pixel_domain_mismatch')
    fail(doc.get('purpose') in {'engineering_replay','camera_candidate'},'unknown_input_purpose')
    bound={}
    for key in ['source_video','decoded','feature_manifest','segments','physical_review']:
        bound[key]=checked_file(doc.get(key),base)
    fail(doc['source_video']['sha256']==subject['video_sha256'],'video_identity_mismatch')
    decoded=read(bound['decoded']);rows=decoded['records']
    fail(decoded.get('video_sha256')==subject['video_sha256'],'decoded_video_mismatch')
    fail(len(rows)==subject['selected_frames'],'frame_count_mismatch')
    fail(decoded.get('decoded_total')==subject['source_frames'],'source_frame_count_mismatch')
    expected=list(range(0,subject['source_frames'],5))
    fail([r['source_frame_index'] for r in rows]==expected,'registered_frame_roster_mismatch')
    fail(all(r['wh']==subject['source_wh'] for r in rows),'native_raster_mismatch')
    fm=read(bound['feature_manifest'])['frames'];ss=read(bound['segments'])['records'];review=read(bound['physical_review'])
    fail(len(fm)==len(rows),'feature_frame_count_mismatch')
    feature_root=Path(doc['feature_root']);feature_root=feature_root if feature_root.is_absolute() else base/feature_root
    frames=[];xy=[]
    for slot,(r,f) in enumerate(zip(rows,fm)):
        fail(f['slot']==slot and f['source_frame']==r['source_frame_index'],'slot_source_frame_mismatch')
        fail(f['source_sha256']==r['source_sha256'],'image_record_identity_mismatch')
        image=checked_file(dict(path=f['image'],sha256=f['source_sha256']),bound['feature_manifest'].parent)
        fail(image.name==r['name'],'decoded_image_name_mismatch')
        name=f['feature_file'];fail(Path(name).name==name,'feature_filename_not_leaf')
        path=checked_file(dict(path=str(feature_root/name),sha256=f['sha256']),base)
        with np.load(path,allow_pickle=False) as data:a=np.array(data['xy'],dtype=np.float64,copy=True)
        fail(a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all(),'invalid_native_features')
        a.setflags(write=False);xy.append(a)
        frames.append(dict(slot=slot,source_frame=f['source_frame'],image=str(image),image_sha256=f['source_sha256'],feature_sha256=f['sha256']))
    segments={};family_splits={};node_owners={}
    for s in ss:
        sid=s['id'];fail(sid not in segments,'duplicate_segment_identity')
        fail(s['split'] in {'FIT','HELD'},'partition_not_original_FIT_HELD')
        fail(type(s['development_only']) is bool,'development_flag_missing')
        family_splits.setdefault(s['family_id'],set()).add(s['split'])
        fail(len(family_splits[s['family_id']])==1,'family_split_leak')
        nodes=[tuple(n) for n in s['nodes']]
        fail(bool(nodes) and len(nodes)==len(set(nodes)),'duplicate_or_empty_nodes')
        for node in nodes:
            i,k=node
            fail(type(i) is int and type(k) is int and 0<=i<len(frames) and 0<=k<len(xy[i]),'invalid_original_node')
            fail(node not in node_owners,'node_in_multiple_segments')
            node_owners[node]=sid
        segments[sid]=s
    reviewed={};conflicts=set()
    fail(review.get('actual_saved_pixels_reviewed') is True,'physical_review_not_performed')
    for rr in review['rows']:
        sid=rr['segment_id'];fail(sid in segments,'review_segment_missing')
        s=segments[sid]
        fail(rr['family_id']==s['family_id'] and rr['split']==s['split'],'review_original_identity_mismatch')
        fail(rr['development_only']==s['development_only'],'review_development_mismatch')
        fail(sid not in reviewed,'duplicate_review_for_segment')
        observed=set()
        for v in rr['actual_saved_views_reviewed']:
            node=(v['slot'],v['keypoint']);fail(list(node) in s['nodes'],'review_node_not_in_segment')
            fail(v['source_frame']==frames[node[0]]['source_frame'],'review_source_frame_mismatch')
            fail(np.array_equal(np.asarray(v['uv'],dtype=np.float64),xy[node[0]][node[1]]),'review_native_pixel_changed')
            observed.add(node)
        if rr['classification']=='confirmed_physical_identity_contradiction':
            fail(len(observed)>=2 and bool(rr.get('note')),'contradiction_without_observation_evidence')
            conflicts.add(s['family_id'])
        reviewed[sid]=(rr,observed)
    tracks=[];width,height=subject['source_wh']
    for s in ss:
        rr,observed=reviewed.get(s['id'],({},set()))
        classification=rr.get('classification','unreviewed')
        role='unknown'
        if classification=='confirmed_static_background_material':role='static_background'
        elif classification=='confirmed_person_material':role='person'
        reasons=[]
        if s['split']!='FIT':reasons.append('heldout_cannot_fit')
        if s['development_only']:reasons.append('development_only')
        if s['family_id'] in conflicts:reasons.append('family_physical_identity_conflict')
        if s.get('topologically_eligible') is not True or len(s['nodes'])<3:reasons.append('insufficient_frozen_topology')
        if len({i for i,k in s['nodes']})!=len(s['nodes']):reasons.append('ambiguous_same_frame_nodes')
        if role!='static_background':reasons.append('role_'+role)
        if rr.get('exact_material_accepted') is not True:reasons.append('physical_material_unknown')
        if rr.get('staticity_verified') is not True:reasons.append('staticity_unknown')
        if rr.get('all_nodes_actually_reviewed') is not True or observed!={tuple(n) for n in s['nodes']}:
            reasons.append('incomplete_node_review')
        uncertainty=rr.get('localization_uncertainty_px')
        if not isinstance(uncertainty,(int,float)) or isinstance(uncertainty,bool) or not math.isfinite(uncertainty) or uncertainty<0:
            reasons.append('measurement_uncertainty_unknown')
        claims=rr.get('supporting_evidence',[])
        if not claims:reasons.append('physical_evidence_reference_missing')
        else:
            for ref in claims:
                evidence_path=checked_file(ref,bound['physical_review'].parent)
                if role=='static_background' and rr.get('exact_material_accepted') is True:
                    check_material_support(evidence_path,subject['video_sha256'],s,frames,xy,role)
        observations=[]
        for i,k in s['nodes']:
            f=frames[i];uv=tuple(float(v) for v in xy[i][k])
            if not (0<=uv[0]<width and 0<=uv[1]<height):reasons.append('outside_native_image')
            observations.append(NativeObservation(i,f['source_frame'],k,f['image_sha256'],f['feature_sha256'],uv))
        tracks.append(FrozenTrack(s['id'],s['family_id'],s['split'],s['development_only'],tuple(observations),role,s['family_id'] in conflicts,tuple(dict.fromkeys(reasons))))
    return FrozenInput(expected_sha256,bound['source_video'],subject['video_sha256'],subject['id'],doc['purpose'],decoded,tuple(frames),tuple(xy),tuple(tracks),tuple((k,str(v),sha(v)) for k,v in bound.items()))

def solver_tracks(bundle):
    """Select admissible FIT only. Every original HELD stays separately labelled."""
    fit=bundle.require_fit_support()
    fail(bundle.purpose=='camera_candidate','engineering_replay_not_a_camera_candidate')
    def record(t):
        return dict(id=t.segment_id,family_id=t.family_id,held=t.partition=='HELD',
            original_partition=t.partition,development_only=t.development_only,
            material_role=t.material_role,family_conflict=t.family_conflict,
            nodes=[(o.slot,o.keypoint_id) for o in t.observations])
    return [record(t) for t in fit],[record(t) for t in bundle.held_tracks]
