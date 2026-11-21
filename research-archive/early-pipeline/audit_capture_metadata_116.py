from pathlib import Path
import subprocess, json, hashlib, struct, collections, math, datetime

B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O = B/'current/portrait-alignment-open-source-116'
VIDEO = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4')
PROBE = 'C:/Users/Administrator/ffmpeg/bin/ffprobe.exe'
O.mkdir(parents=True, exist_ok=True)
def save(name, obj):
    p=O/name; t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8'); t.replace(p)
def probe(*args):
    return json.loads(subprocess.run([PROBE,'-v','error',*args,'-print_format','json',str(VIDEO)],check=True,capture_output=True,text=True,encoding='utf8').stdout)
def varint(data, pos):
    value=0
    for shift in range(0,70,7):
        if pos>=len(data): raise ValueError('truncated varint')
        b=data[pos];pos+=1;value|=(b&127)<<shift
        if b<128:return value,pos
    raise ValueError('overlong varint')
def fields(data, depth=0):
    if depth>12:raise ValueError('nesting')
    result=[];pos=0
    while pos<len(data):
        key,pos=varint(data,pos);tag=key>>3;wire=key&7
        if tag==0 or tag>536870911:raise ValueError('invalid tag')
        f={'field':tag,'wire_type':wire}
        if wire==0:f['uint'],pos=varint(data,pos)
        elif wire in (1,5):
            n=8 if wire==1 else 4
            if pos+n>len(data):raise ValueError('truncated fixed')
            b=data[pos:pos+n];pos+=n
            f['uint']=int.from_bytes(b,'little')
            x=struct.unpack('<d' if n==8 else '<f',b)[0]
            f['float_interpretation']=x if math.isfinite(x) else None
        elif wire==2:
            n,pos=varint(data,pos)
            if pos+n>len(data):raise ValueError('truncated length')
            b=data[pos:pos+n];pos+=n;f['length']=n
            try:
                text=b.decode('utf8')
                if text and all(c.isprintable() for c in text):f['text']=text
            except UnicodeDecodeError:pass
            if 'text' not in f:
                try:f['nested']=fields(b,depth+1)
                except ValueError:f['opaque_hex']=b.hex()
        else:raise ValueError(f'unsupported wire type {wire}')
        result.append(f)
    return result
def collect(fs,prefix='',out=None):
    out=[] if out is None else out
    for f in fs:
        key=prefix+str(f['field'])
        if 'nested' in f:collect(f['nested'],key+'.',out)
        else:out.append((key,f))
    return out
def boxes(data, start, end, prefix=''):
    out=[];p=start
    containers={'moov','trak','mdia','minf','stbl','edts','udta','meta','ilst','dinf','mvex'}
    while p+8<=end:
        n=int.from_bytes(data[p:p+4],'big');typ=data[p+4:p+8].decode('latin1');head=8
        if n==1:
            if p+16>end:break
            n=int.from_bytes(data[p+8:p+16],'big');head=16
        elif n==0:n=end-p
        if n<head or p+n>end:break
        path=prefix+'/'+typ
        out.append({'box':path,'position':p,'size':n})
        if typ in containers:out+=boxes(data,p+head+(4 if typ=='meta' else 0),p+n,path)
        p+=n
    return out

before=hashlib.sha256(VIDEO.read_bytes()).hexdigest()
metadata=probe('-show_format','-show_streams');save('video-ffprobe.json',metadata)
packets=probe('-select_streams','d:0','-show_packets')['packets']
video_packets=probe('-select_streams','v:0','-show_packets')['packets']
decoded=[];errors=[];stats=collections.defaultdict(list)
with VIDEO.open('rb') as file:
    for i,p in enumerate(packets):
        file.seek(int(p['pos']));data=file.read(int(p['size']))
        row={k:p.get(k) for k in ['pts','pts_time','duration','size','pos']};row['packet_index']=i;row['sha256']=hashlib.sha256(data).hexdigest()
        try:
            row['wire_fields']=fields(data)
            for key,f in collect(row['wire_fields']):stats[key].append(f)
        except ValueError as e:errors.append({'packet_index':i,'error':str(e)});row['parse_error']=str(e)
        decoded.append(row)
save('hovr-wire-records.json',decoded)
field_summary=[]
for path,fs in sorted(stats.items()):
    row={'field_path':path,'count':len(fs),'wire_types':sorted(set(f['wire_type'] for f in fs))}
    for attr in ['uint','float_interpretation','text','opaque_hex']:
        values=[f[attr] for f in fs if attr in f and f[attr] is not None]
        if not values:continue
        if attr in ('uint','float_interpretation'):row[attr]={'min':min(values),'max':max(values),'first':values[0],'last':values[-1]}
        else:row[attr]={'unique_count':len(set(values)),'examples':list(dict.fromkeys(values))[:3]}
    field_summary.append(row)
save('hovr-wire-summary.json',{'semantics':'Wire-format parse only; field names, units, coordinate frames and camera meaning are NOT inferred from values. Not an official schema decoder.','parse_errors':errors,'fields':field_summary})
inventory=boxes(VIDEO.read_bytes(),0,VIDEO.stat().st_size);save('mp4-boxes.json',inventory)
vp=[float(p['pts_time']) for p in video_packets]
timing=[]
for i,p in enumerate(packets):
    t=float(p['pts_time']);j=min(range(len(vp)),key=lambda j:abs(vp[j]-t))
    timing.append({'metadata_packet':i,'metadata_pts_s':t,'nearest_video_packet_index':j,'video_pts_s':vp[j],'difference_s':t-vp[j]})
save('metadata-video-timing.json',timing)
frame_records=[]
for row in decoded:
    flat=dict(collect(row.get('wire_fields',[])))
    if '3.1.1' in flat:
        frame_records.append({'packet_index':row['packet_index'],'fixed64_clock':flat['3.1.1']['uint'],
                              'fixed64_index':flat.get('3.1.2',{}).get('uint',0),'metadata_pts':row['pts']})
clock_start=frame_records[0]['fixed64_clock']
index_sequence=[r['fixed64_index'] for r in frame_records]
clock_differences=[(r['fixed64_clock']-clock_start)-int(p['pts']) for r,p in zip(frame_records,video_packets)]
sync={'repeated_records':len(frame_records),'video_packets':len(video_packets),
      'field_3_1_2_matches_0_based_sequence':index_sequence==list(range(len(video_packets))),
      'first_record_field_3_1_2_absent_interpreted_as_zero_for_this_test':True,
      'field_3_1_1_minus_first_vs_video_pts_max_abs_ticks':max(abs(x) for x in clock_differences),
      'video_time_base':metadata['streams'][0]['time_base'],
      'metadata_packet_pts_offset_from_video_max_ticks':max(abs(r['metadata_pts']-int(p['pts'])) for r,p in zip(frame_records,video_packets)),
      'interpretation':'The numeric sequence and relative clocks agree with frame numbering/timing. This does not decode the remaining fields or confirm calibration.'}
save('hovr-frame-timing-check.json',sync)
summary={'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'video':str(VIDEO),'video_sha256':before,
    'source_unchanged':before==hashlib.sha256(VIDEO.read_bytes()).hexdigest(),
    'video_frame_count':len(video_packets),'metadata_packets':len(packets),'metadata_payload_bytes':sum(int(p['size']) for p in packets),
    'device':metadata['format']['tags'].get('device_model'),'schema_tag':metadata['format']['tags'].get('hovr_schema_version'),
    'metadata_codec':'hovr','wire_parse_complete':not errors,'official_schema_available':False,
    'timing_check':sync,
    'original_capture_intrinsics_confirmed':False,'original_capture_extrinsics_confirmed':False,
    'status':'capture_metadata_found_but_camera_semantics_unresolved',
    'current_074_camera_source':'COLMAP SfM and bundle adjustment, NOT verified capture-time calibration',
    'known_source_tags':{k:metadata['format']['tags'].get(k) for k in ['device_model','hovr_schema_version','name','TrajectoryMode','MCTF','EffectMode']},
    'limits':['The hovr track is real device-origin metadata; its existence does not prove that K, distortion or R/t are stored.',
              '179 metadata records cannot be assigned to 177 video frames by list index; inspect message type and timestamps.',
              'No official field schema has been found yet. Private board/media identifiers stay in local raw evidence, not published summaries.',
              'This audit did not connect to computer 00 or inspect its capture-device export directory.'],
    'new_training_started':False}
save('capture-provenance.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('WIRE_FIELD_STATISTICS')
for row in field_summary:
    if row['count']>2:print(json.dumps(row,ensure_ascii=False))
