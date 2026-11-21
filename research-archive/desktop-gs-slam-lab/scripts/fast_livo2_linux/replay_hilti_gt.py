#!/usr/bin/env python3
"""Prepare one Hilti exp14 LIO/LIVO replay; default is read-only preflight.

With explicit --execute, starts only a private master and owned process groups. It does not bootstrap,
materialize sources, recompile, recalibrate, or evaluate ground-truth accuracy.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import xmlrpc.client

EXPECTED_BYTES = 6260771085
EXPECTED_SHA = '18eea013076b8263ababf4f36eb68d331564a0a493262b006dc0dddc2aa85b82'
COMMIT = '0d2c0346107b75b59934975adec9a6eeeb913c64'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def inventory(root):
    result = {}
    if root.exists():
        for file in sorted(root.rglob('*')):
            if file.is_symlink():
                raise RuntimeError('Refusing a symlink in runtime Log: ' + str(file))
            if file.is_file():
                result[str(file.relative_to(root))] = {'bytes': file.stat().st_size,
                    'sha256': digest(file)}
    return result


def pinned_files(package, receipt, override_receipt=None):
    record = json.loads(receipt.read_text(encoding='utf-8'))
    if record.get('commit') != COMMIT:
        raise RuntimeError('Unexpected source acquisition commit')
    overrides = {}
    if override_receipt:
        override_record = json.loads(override_receipt.read_text(encoding='utf-8'))
        if override_record.get('source_commit') != COMMIT:
            raise RuntimeError('Source override has a different baseline commit')
        for row in override_record['files']:
            name = row['path']
            if name in overrides:
                raise RuntimeError('Duplicate source override: ' + name)
            overrides[name] = row
        official_names = {row['path'] for row in record['files']}
        if not overrides or not set(overrides).issubset(official_names):
            raise RuntimeError('Overrides must explicitly bind existing pinned source files')
    checks = []
    for row in record['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Invalid relative source path')
        file = package / rel
        actual = digest(file)
        override = overrides.get(row['path'])
        if override and override['original_sha256'] != row['sha256']:
            raise RuntimeError('Override baseline hash mismatch: ' + str(rel))
        wanted_sha = override['patched_sha256'] if override else row['sha256']
        wanted_bytes = override['patched_bytes'] if override else row['bytes']
        if actual != wanted_sha or file.stat().st_size != wanted_bytes:
            raise RuntimeError('Pinned source mismatch: ' + str(rel))
        checks.append({'path': str(rel), 'bytes': file.stat().st_size, 'sha256': actual,
            'official_sha256': row['sha256'], 'explicit_local_patch': bool(override)})
    return {'commit': COMMIT, 'selected_receipt_files_verified': len(checks), 'files': checks,
        'source_override_receipt_sha256': digest(override_receipt) if override_receipt else None,
        'local_patched_files': sorted(overrides),
        'scope': 'All pinned files checked against official hashes or an explicit reviewed local patch receipt. Runtime Log outputs inventoried separately.'}


def child_exit_summary(out, fusion_pid):
    """Read actual roslaunch child lifecycle records, independently of parent exit."""
    rows = {name: {'start_pids': [], 'exit_codes': [], 'terminal_evidence': []}
        for name in ['laserMapping-1']}
    for file in sorted((out / 'ros-node-logs').glob('*/roslaunch-*.log')):
        if file.is_symlink() or file.parent.is_symlink():
            continue
        for number, line in enumerate(file.read_text(errors='replace').splitlines(), 1):
            match = re.search(r'process\[([^]]+)\]: started with pid \[(\d+)\]', line)
            if match and match.group(1) in rows:
                rows[match.group(1)]['start_pids'].append(int(match.group(2)))
            match = re.search(r'process\[([^]]+)\]: SIGINT killed with return value (-?\d+)', line)
            if match and match.group(1) in rows:
                row = rows[match.group(1)]
                row['exit_codes'].append(int(match.group(2)))
                row['terminal_evidence'].append({'file': str(file.relative_to(out)),
                    'line': number, 'text': line, 'file_sha256': digest(file)})
    for row in rows.values():
        row['verified_exit_zero'] = bool(row['start_pids'] and row['exit_codes']) and all(code == 0 for code in row['exit_codes'])
    pid_matches = rows['laserMapping-1']['start_pids'] == [fusion_pid]
    return {'children': rows, 'fusion_pid_matches_actual_started_child': pid_matches,
        'all_required_children_exit_zero': pid_matches and all(row['verified_exit_zero'] for row in rows.values()),
        'scope': 'Missing terminal child evidence is unverified and fails acceptance; parent roslaunch exit 0 cannot substitute.'}


def ros_environment(root):
    setups = [Path('/opt/ros/noetic/setup.bash'), root / 'catkin_ws/devel/setup.bash']
    if not all(p.is_file() for p in setups):
        raise RuntimeError('ROS or compiled catkin setup is missing; do not bootstrap from this runner')
    result = subprocess.run(['bash', '-c', 'source "$1" && source "$2" && env -0',
        'replay-env', str(setups[0]), str(setups[1])], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True, timeout=30)
    return dict(item.split('=', 1) for item in result.stdout.decode().split('\0')
        if '=' in item)


class TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=2)


def master(uri):
    return xmlrpc.client.ServerProxy(uri, transport=TimeoutTransport(), allow_none=True)


def master_ready(uri):
    try:
        response = master(uri).getPid('/gs_replay_preflight')
        return response[0] == 1
    except (OSError, xmlrpc.client.Error):
        return False


def mapper_alive(uri):
    try:
        response = master(uri).lookupNode('/gs_replay_probe', '/laserMapping')
        if response[0] != 1:
            return False
        return master(response[2]).getPid('/gs_replay_probe')[0] == 1
    except (OSError, xmlrpc.client.Error):
        return False


def launch_ready(uri):
    try:
        graph = master(uri).getSystemState('/gs_replay_preflight')
        if graph[0] != 1 or not mapper_alive(uri):
            return False
        subscribers = dict(graph[2][1])
        return all('/laserMapping' in subscribers.get(topic, []) for topic in
            ['/hesai/pandar', '/alphasense/imu', '/alphasense/cam0/image_raw'])
    except (OSError, xmlrpc.client.Error):
        return False


class OwnedProcesses:
    def __init__(self, out, env):
        self.out, self.env = out, env
        self.items, self.actions = [], []

    def start(self, label, command):
        file = (self.out / (label + '.log')).open('wb')
        process = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT,
            env=self.env, start_new_session=True)
        self.items.append({'label': label, 'command': command, 'process': process,
            'file': file, 'pid': process.pid, 'pgid': process.pid, 'started_utc': utc()})
        return process

    def stop(self, process):
        if process.poll() is None:
            for sig, timeout in [(signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)]:
                if process.poll() is not None:
                    break
                # Every target is a session/group created by start_new_session above.
                self.actions.append({'owned_pgid': process.pid, 'signal': sig.name, 'utc': utc()})
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue

    def finish(self):
        for row in reversed(self.items):
            self.stop(row['process'])
            row['file'].close()
        return [{'label': row['label'], 'command': row['command'], 'pid': row['pid'],
            'pgid': row['pgid'], 'started_utc': row['started_utc'],
            'actual_exit_code': row['process'].poll()} for row in self.items]


class Monitor:
    """Observe independent ROS subscriptions. Counts never prove estimator drop freedom."""
    def __init__(self, out):
        import rospy
        import numpy as np
        from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2
        from nav_msgs.msg import Odometry, Path as RosPath
        from rosgraph_msgs.msg import Clock
        self.rospy, self.np, self.out = rospy, np, out
        self.lock = threading.Lock()
        self.states, self.headers, self.errors, self.last_cloud = {}, {}, [], {}
        self.last_output_wall = time.monotonic()
        self.sensor_file = (out / 'observed-sensor-headers.csv').open('w', newline='')
        self.sensor_writer = csv.writer(self.sensor_file)
        self.sensor_writer.writerow(['topic','header_seq','header_stamp_ns','received_monotonic_s'])
        self.odom_file = (out / 'observed-odometry.csv').open('w', newline='')
        self.odom_writer = csv.writer(self.odom_file)
        self.odom_writer.writerow(['ros_header_stamp_ns','received_monotonic_s',
            'px','py','pz','qx','qy','qz','qw','finite'])
        rospy.init_node('gs_lab_replay_observer', anonymous=True, disable_signals=True)
        types = {'/alphasense/imu': Imu, '/hesai/pandar': PointCloud2,
            '/alphasense/cam0/image_raw': Image,
            '/rgb_img': Image, '/aft_mapped_to_init': Odometry, '/path': RosPath,
            '/cloud_registered': PointCloud2, '/cloud_visual_sub_map_before': PointCloud2,
            '/Laser_map': PointCloud2, '/LIVO2/imu_propagate': Odometry, '/clock': Clock}
        self.subscribers = [rospy.Subscriber(topic, cls, self.callback, callback_args=topic,
            queue_size=2000 if topic == '/alphasense/imu' else 100,
            buff_size=32 * 1024 * 1024, tcp_nodelay=True) for topic, cls in types.items()]

    def track(self, topic, seq, stamp):
        state = self.states.setdefault(topic, {'count': 0, 'negative_header_steps': 0,
            'duplicate_headers': 0, 'invalid_messages': 0})
        state['count'] += 1
        if stamp is not None:
            key = (seq, stamp)
            known = self.headers.setdefault(topic, set())
            state['duplicate_headers'] += int(key in known)
            known.add(key)
            if 'last_header_stamp_ns' in state:
                state['negative_header_steps'] += int(stamp < state['last_header_stamp_ns'])
            else:
                state['first_header_stamp_ns'] = stamp
            state['last_header_stamp_ns'] = stamp
        return state

    def cloud_xyz(self, message):
        if message.width * message.height == 0:
            return self.np.empty((0, 3), dtype=self.np.float32)
        fields = {field.name: field for field in message.fields}
        arrays = []
        for name in ['x','y','z']:
            field = fields[name]
            if field.datatype not in (7, 8) or field.count != 1:
                raise ValueError('Unexpected XYZ field encoding')
            dtype = ('>' if message.is_bigendian else '<') + ('f4' if field.datatype == 7 else 'f8')
            arrays.append(self.np.ndarray((message.height, message.width), dtype=dtype,
                buffer=message.data, offset=field.offset,
                strides=(message.row_step, message.point_step)).reshape(-1))
        return self.np.column_stack(arrays)

    def callback(self, message, topic):
        try:
            if topic == '/clock':
                seq, stamp = None, message.clock.to_nsec()
            else:
                seq, stamp = message.header.seq, message.header.stamp.to_nsec()
            now = time.monotonic()
            with self.lock:
                state = self.track(topic, seq, stamp)
                if topic in ['/hesai/pandar','/alphasense/imu','/alphasense/cam0/image_raw']:
                    self.sensor_writer.writerow([topic, seq, stamp, now])
                if topic == '/alphasense/imu':
                    values = [getattr(vector, axis) for vector in
                        [message.angular_velocity, message.linear_acceleration] for axis in ['x','y','z']]
                    state['invalid_messages'] += int(not all(math.isfinite(x) for x in values))
                elif topic in ['/alphasense/cam0/image_raw','/rgb_img']:
                    valid = message.width > 0 and message.height > 0 and len(message.data) == message.step * message.height
                    state['invalid_messages'] += int(not valid)
                    state['last_shape'] = [message.height, message.width]
                    state['last_encoding'] = message.encoding
                elif topic in ['/aft_mapped_to_init','/LIVO2/imu_propagate']:
                    pose = message.pose.pose
                    values = [pose.position.x, pose.position.y, pose.position.z,
                        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    finite = all(math.isfinite(x) for x in values)
                    state['invalid_messages'] += int(not finite)
                    self.odom_writer.writerow([stamp, now] + values + [finite])
                    self.last_output_wall = now
                elif topic in ['/cloud_registered','/cloud_visual_sub_map_before','/Laser_map']:
                    xyz = self.cloud_xyz(message)
                    mask = self.np.isfinite(xyz).all(axis=1)
                    state['last_point_count'] = int(len(xyz))
                    state['nonfinite_xyz_points'] = state.get('nonfinite_xyz_points', 0) + int((~mask).sum())
                    state['total_xyz_points_observed'] = state.get('total_xyz_points_observed', 0) + int(len(xyz))
                    state['invalid_messages'] += int(not bool(mask.all()))
                    state['last_frame_id'] = message.header.frame_id
                    self.last_cloud[topic] = xyz[mask].astype(self.np.float32, copy=True)
                    self.last_output_wall = now
                elif topic == '/path':
                    state['last_pose_count'] = len(message.poses)
                    if message.poses:
                        pose = message.poses[-1].pose
                        state['invalid_last_pose'] = not all(math.isfinite(x) for x in
                            [pose.position.x, pose.position.y, pose.position.z,
                             pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        except Exception as error:
            with self.lock:
                self.errors.append({'topic': topic, 'error': repr(error), 'utc': utc()})

    def save(self, expected):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.rospy.signal_shutdown('Owned observer finished; private master cleanup is separate')
        with self.lock:
            self.sensor_file.close()
            self.odom_file.close()
            states = dict(self.states)
            comparisons = {}
            for topic, count in expected.items():
                observed = states.get(topic, {}).get('count', 0)
                comparisons[topic] = {'bag_index_messages': count, 'observer_messages': observed,
                    'observer_count_difference': observed-count,
                    'scope': 'Independent observer reception, not estimator callback/processed-message counts.'}
            pairing = {'raw_topic':'/alphasense/cam0/image_raw', 'source_encoding':'mono8', 'republisher_required':False,
                'conversion':'official GetImage uses cv_bridge to bgr8', 'scope':'Observer reception only, not estimator image acceptance.'}
            for topic, xyz in self.last_cloud.items():
                filename = topic.strip('/').replace('/', '_') + '-last-observed-xyz.npy'
                self.np.save(str(self.out / filename), xyz, allow_pickle=False)
            record = {'topics': states, 'bag_vs_observer': comparisons,
                'raw_image_transport': pairing, 'callback_errors': self.errors,
                'processed_message_drop_freedom_proven': False,
                'clock_note': 'Odometry/map headers use upstream ros::Time::now(), simulated clock here; use sensor-time trajectory files for analysis.',
                'cloud_note': 'Saved arrays are last observed registered scan / visual sub-map, not a complete accumulated global map.',
                'imu_note': 'Finite raw numeric values only; source units are not assumed to be SI.'}
            write_json(self.out / 'topic-observation.json', record)
            return record


def trajectory_summary(path, phase):
    rows, bad, negative, previous, norms = 0, [], 0, None, []
    first, last = None, None
    if not path.exists():
        return {'exists': False, 'phase': phase, 'rows': 0}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        values = line.split()
        try:
            values = [float(x) for x in values]
            if len(values) != 8 or not all(math.isfinite(x) for x in values):
                raise ValueError('Expected eight finite TUM columns')
            t = values[0]
            if previous is not None:
                negative += int(t < previous)
            first = t if first is None else first
            last, previous = t, t
            norms.append(math.sqrt(sum(x*x for x in values[4:8])))
            rows += 1
        except ValueError as error:
            bad.append({'line': number, 'error': str(error)})
    return {'exists': True, 'phase': phase, 'rows': rows, 'invalid_rows': bad,
        'negative_timestamp_steps': negative, 'first_sensor_timestamp_s': first,
        'last_sensor_timestamp_s': last,
        'quaternion_norm_min': min(norms) if norms else None,
        'quaternion_norm_max': max(norms) if norms else None,
        'all_quaternions_unit_within_1e_3': bool(norms) and all(abs(x-1.0) <= 1e-3 for x in norms),
        'sha256': digest(path), 'bytes': path.stat().st_size,
        'orientation_ground_truth': False, 'trajectory_accuracy_evaluated': False}


def visual_log_summary(log):
    text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace')) if log.exists() else ''
    markers = {}
    for name, pattern in [
            ('handle_vio_raw_features', r'\[ VIO \] Raw feature num: (\d+)'),
            ('retrieved_visual_candidates', r'\[ VIO \] Retrieve (\d+) points from visual sparse map'),
            ('appended_visual_map_points', r'\[ VIO \] Append (\d+) new visual map points'),
            ('updated_visual_map_points', r'\[ VIO \] Update (\d+) points in visual submap')]:
        values = [int(x) for x in re.findall(pattern, text)]
        markers[name] = {'lines': len(values), 'nonzero_lines': sum(x > 0 for x in values),
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'sum': sum(values), 'values_in_log_order': values}
    markers['handle_vio_no_points_lines'] = text.count('[ VIO ] No point!!!')
    markers['lio_map_update_lines'] = text.count('[ LIO ] Update Voxel Map')
    markers['imu_loopback_reports'] = text.count('imu loop back, offset:')
    markers['data_cut_thrown_image_reports'] = text.count('[ Data Cut ] Throw one image frame!')
    for name, label in [('jacobian_stage', '-> computeJacobian'), ('ekf_stage', '-> updateEKF')]:
        values = [float(x) for x in re.findall(r'\|\s*' + re.escape(label) + r'\s*\|\s*([0-9.eE+-]+)\s*\|', text)]
        markers[name] = {'timing_lines': len(values), 'positive_timing_lines': sum(x > 0 for x in values),
            'scope': 'Existing stage timing prints; not a residual log or an accepted-step counter.'}
    return {'source_commit': COMMIT, 'markers': markers,
        'retained_candidate_scope': 'Retrieve N is total_points after visual-map retrieval/outlier filtering. Nonzero N shows candidate constraints reach the update stage, not successful acceptance of every EKF step.',
        'visual_update_acceptance_proven_by_these_markers': False,
        'residual_values_exported_by_upstream_log': False,
        'state_change_evidence': 'Actual Log/mat_pre.txt and mat_out.txt are archived for separate per-VIO pre/post-state analysis. RGB and image_poses alone prove output calls, not effective visual constraints.'}


def preflight_only(args, lab, root, out, run_name):
    """No master/node/player starts, no compile, no official runtime Log writes."""
    import yaml
    from xml.sax.saxutils import escape
    package = root/'catkin_ws/src/FAST-LIVO2'
    build_path = lab/'outputs/fast-livo2-shutdown-fix-001/build-after-verification.json'
    build = json.loads(build_path.read_text())
    source = pinned_files(package, lab/'outputs/fast-livo2-source-audit-001/repository-acquisition.json', args.source_override_receipt)
    write_json(out/'native-source-preflight.json', source)
    binary = root/'catkin_ws/devel/lib/fast_livo/fastlivo_mapping'
    compiled = [{'path':str(binary),'sha256':digest(binary),'expected_sha256':build['binary_after_sha256']}]
    for item in build['libraries_after']:
        p=Path(item['path'])
        compiled.append({'path':str(p),'sha256':digest(p),'expected_sha256':item['sha256']})
    checks={'source_75_pinned_or_explicit_CMake_override':source['selected_receipt_files_verified']==75,
            'accepted_binary_and_all_algorithm_libraries':all(x['sha256']==x['expected_sha256'] for x in compiled)}
    nm=subprocess.run(['nm','-D','-C',str(root/'catkin_ws/devel/lib/libpre.so')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    symbols=[line for line in nm.stdout.decode(errors='replace').splitlines() if 'Preprocess::xt32_handler' in line or 'Preprocess::process' in line]
    checks['XT32_handler_symbol_in_actual_compiled_libpre']=nm.returncode==0 and any('Preprocess::xt32_handler' in x for x in symbols)
    sensor_bag=Path('/opt/gs-slam-lab/fast-livo2-data/hilti2022-recovery-003-exp14/rosbags/exp14_basement_2.bag')
    checks['actual_complete_bag_official_size_SHA256']=sensor_bag.stat().st_size==EXPECTED_BYTES and digest(sensor_bag)==EXPECTED_SHA
    recovery=lab/'outputs/fast-livo2-gt-input-recovery-003'
    frozen=json.loads((recovery/'receipt.json').read_text())
    audit_key='offline-audit-runs/20261010T094728Z_ae2afedf/receipt.json'
    audit_binding=next(v for k,v in frozen['evidence'].items() if k.replace('\\','/')==audit_key)
    audit_path=recovery/audit_binding['path'].replace('\\','/')
    checks['frozen_complete_input_audit_SHA256_and_14_checks']=digest(audit_path)==audit_binding['sha256'] and frozen['checks_passed']==14 and frozen['ready_for_future_replay']
    audit=json.loads(audit_path.read_text())
    data=sensor_bag.parent.parent
    input_bindings=[]
    for name in ['ground_truth/exp14_basement_2_imu.txt','ground_truth/exp14_basement_2_imu_3dof.txt']:
        p=data/name
        item=audit['small_files_before'][str(p)]
        input_bindings.append({'path':str(p),'sha256':digest(p),'expected_sha256':item['sha256']})
    checks['both_frozen_GT_files_unchanged']=all(x['sha256']==x['expected_sha256'] for x in input_bindings)
    conf=lab/('configs/fast-livo2-hilti-local/HILTI22.'+args.mode+'.upstream-zero.yaml')
    camera=lab/'configs/fast-livo2-hilti-local/camera_fisheye_HILTI22.yaml'
    c=yaml.safe_load(conf.read_text()); base=yaml.safe_load((package/'config/HILTI22.yaml').read_text())
    wanted=json.loads(json.dumps(base));wanted['common']['img_en']=int(args.mode=='livo')
    wanted['evo']['seq_name']='exp14_basement_2_'+args.mode+'_REPLACE_WITH_UNIQUE_RUN_ID';wanted['evo']['pose_output_en']=True
    wanted['image_save']['img_save_en']=args.mode=='livo';wanted['image_save']['interval']=20
    wanted['pcd_save']['pcd_save_en']=False;wanted['pcd_save']['colmap_output_en']=False
    checks['prepared_config_only_registered_mode_and_output_controls']=c==wanted
    checks['camera_yaml_byte_identical_to_fixed_official_source']=digest(camera)==digest(package/'config/camera_fisheye_HILTI22.yaml')
    env=ros_environment(root)
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(args.port),ROS_IP='127.0.0.1',ROS_HOME=str(out/'ros-home'),ROS_LOG_DIR=str(out/'parse-logs'))
    env.pop('ROS_HOSTNAME',None)
    for path in reversed(env.get('PYTHONPATH','').split(':')):
        if path:sys.path.insert(0,path)
    import rosbag
    from cv_bridge import CvBridge
    with rosbag.Bag(str(sensor_bag),'r') as bag:
        info=bag.get_type_and_topic_info()
        types={'/hesai/pandar':'sensor_msgs/PointCloud2','/alphasense/imu':'sensor_msgs/Imu','/alphasense/cam0/image_raw':'sensor_msgs/Image'}
        checks['native_bag_required_topics_types_and_counts']=all(t in info.topics and info.topics[t].msg_type==typ and info.topics[t].message_count==audit['bag_index'][t]['message_count'] for t,typ in types.items())
        _,image,_=next(bag.read_messages(topics=['/alphasense/cam0/image_raw']))
        bgr=CvBridge().imgmsg_to_cv2(image,'bgr8')
        checks['actual_mono8_to_BGR8_cv_bridge_supported']=image.encoding=='mono8' and list(bgr.shape)==[540,720,3] and str(bgr.dtype)=='uint8'
        bridge={'input_encoding':image.encoding,'actual_converted_shape':list(bgr.shape),'actual_converted_dtype':str(bgr.dtype),
                'republish_needed':False,'official_call':'LIVMapper::GetImage cv_bridge::toCvCopy(img_msg, "bgr8")'}
    wrapper=out/'candidate-run.launch'
    wrapper.write_text('''<launch>
  <param name="use_sim_time" value="true" />
  <rosparam command="load" file="%s" />
  <node pkg="fast_livo" type="fastlivo_mapping" name="laserMapping" output="screen">
    <rosparam file="%s" />
  </node>
  <param name="evo/seq_name" value="%s" type="str" />
  <param name="image_save/interval" value="%d" type="int" />
  <param name="time_offset/lidar_time_offset" value="0.0" type="double" />
</launch>
''' % (escape(str(conf)),escape(str(camera)),run_name,args.image_interval),encoding='utf8')
    parsed=subprocess.run(['roslaunch','--dump-params',str(wrapper)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (out/'planned-parameters.yaml').write_bytes(parsed.stdout)
    checks['roslaunch_parameter_parse_only_no_nodes']=parsed.returncode==0
    params=yaml.safe_load(parsed.stdout) if parsed.returncode==0 else {}
    desired={'/common/img_en':int(args.mode=='livo'),'/common/lidar_en':1,'/imu/imu_en':True,
             '/preprocess/lidar_type':5,'/preprocess/hilti_en':True,'/preprocess/scan_line':32,
             '/common/lid_topic':'/hesai/pandar','/common/imu_topic':'/alphasense/imu','/common/img_topic':'/alphasense/cam0/image_raw',
             '/time_offset/img_time_offset':0.0,'/time_offset/imu_time_offset':0.0,'/time_offset/lidar_time_offset':0.0,
             '/evo/seq_name':run_name,'/image_save/interval':args.image_interval,'/laserMapping/cam_model':'EquidistantCamera',
             '/laserMapping/cam_width':720,'/laserMapping/cam_height':540}
    parameter_checks={k:{'wanted':v,'actual':params.get(k),'passed':params.get(k)==v} for k,v in desired.items()}
    checks['dumped_Hilti_type5_topics_geometry_time_and_unique_output_controls']=all(x['passed'] for x in parameter_checks.values())
    nodes=subprocess.run(['roslaunch','--nodes',str(wrapper)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    (out/'planned-nodes.txt').write_bytes(nodes.stdout)
    checks['one_mapper_node_no_RVIZ_no_republisher']=nodes.returncode==0 and nodes.stdout.decode().split()==['/laserMapping']
    record={'status':'read_only_preflight_passed' if all(checks.values()) else 'preflight_failed','observed_utc':utc(),'checks':checks,
            'mode':args.mode,'ROS_nodes_started':False,'replay_started':False,'accuracy_measured':False,'compiled_files':compiled,
            'accepted_build_receipt_SHA256':digest(build_path),'nm_actual_exit_code':nm.returncode,'actual_compiled_XT32_symbols':symbols,
            'bag':{'path':str(sensor_bag),'bytes':EXPECTED_BYTES,'sha256':EXPECTED_SHA},'GT_inputs':input_bindings,
            'config':{'path':str(conf),'sha256':digest(conf),'camera_path':str(camera),'camera_sha256':digest(camera)},
            'actual_offline_image_conversion':bridge,'parameter_checks':parameter_checks,
            'time_policy':{'img_offset_used_s':0,'imu_offset_used_s':0,'publisher_timeshift_cam_imu_s':0.0019071204237431304,
                           'source_image_formula':'corrected_image_stamp = original_header + img_time_offset',
                           'source_IMU_formula':'corrected_IMU_stamp = original_header - imu_time_offset',
                           'no_GT_offset_search':True,'publisher_shift_sign_exposure_not_assumed_valid':True},
            'future_trajectory_protocol':{'primary':'Log/result/<unique_run_name>.txt: IMU-body pose after LIO, before same-cycle VIO; sensor last_lio_update_time',
               'LIVO_primary_scope':'includes previous visual-update history; not post-current-VIO',
               'supplementary':'LIVO Log/image/image_poses.txt after VIO, sampled by interval '+str(args.image_interval),
               'not_full_40Hz_post_VIO':True,'odometry_header_not_primary':'upstream ros::Time::now() even under simulated clock cannot replace sensor-time trajectory',
               'dense_GT_reference':'689 publisher IMU-frame poses, no lever arm; independent evaluator must use rigid SE3 scale1, fixed time association, no time tuning on GT',
               'sparse_GT':'four publisher IMU-named surveyed positions; suffix branch/physical convention remains diagnostic, not primary score'}}
    write_json(out/'preflight.json',record)
    if not all(checks.values()):raise RuntimeError('Hilti read-only preflight failed: '+str([k for k,v in checks.items() if not v]))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['livo','lio'], default='lio')
    parser.add_argument('--execute', action='store_true', help='Start actual private ROS replay; default performs preflight only')
    parser.add_argument('--windows-lab', type=Path, default=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab'))
    parser.add_argument('--root', type=Path, default=Path('/opt/gs-slam-lab/fast-livo2'))
    parser.add_argument('--port', type=int, default=11321)
    parser.add_argument('--image-interval', type=int, default=20)
    parser.add_argument('--run-group', default='fast-livo2-hilti-gt-001')
    parser.add_argument('--source-override-receipt', type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux' or not (1024 <= args.port <= 65535) or args.image_interval < 1 or not re.fullmatch(r'fast-livo2-hilti-gt-\d{3}', args.run_group):
        parser.error('Requires Linux, an unprivileged private ROS port, and image interval >=1')
    lab, root = args.windows_lab.resolve(), args.root.resolve()
    # rosbag's LZ4 extension needs ROS shared libraries visible at process startup.
    # Source existing setups and replace this interpreter before any ROS import.
    if os.environ.get('GS_HILTI_ENV_READY') != '1':
        startup_env=ros_environment(root)
        startup_env['GS_HILTI_ENV_READY']='1'
        os.execve(sys.executable,[sys.executable]+sys.argv,startup_env)
    run_name = 'exp14_basement_2_' + args.mode + '_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = lab / 'outputs' / args.run_group / ('runs' if args.execute else 'preflight-runs') / run_name
    if args.source_override_receipt is None:
        args.source_override_receipt = lab / 'outputs/fast-livo2-shutdown-fix-001/source-override-receipt.json'
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / 'runner-source.py')
    if args.source_override_receipt:
        shutil.copy2(str(args.source_override_receipt), str(out / 'source-override-receipt.json'))
    state = {'status': 'preflight', 'run_name': run_name, 'mode': args.mode,
        'started_utc': utc(), 'output': str(out), 'source_commit': COMMIT,
        'rate': 0.5, 'ground_truth_available': True, 'accuracy_evaluated': False,
        'GT_scope':'Publisher dense IMU reference, 689 poses; sparse positions 4 checkpoints with filename/frame caveat; no GT enters estimator',
        'time_offset_policy':'upstream-zero baseline; publisher camera shift +1.9071204 ms is recorded but not silently applied',
        'no_processed_message_drop_claim': True, 'fusion_started': False,
        'private_master_uri': 'http://127.0.0.1:' + str(args.port),
        'post_vio_image_interval': args.image_interval if args.mode == 'livo' else None,
        'performance_scope': 'Half-speed functional replay, monitored ROS transport and sampled PNG disk writes; not a throughput benchmark.'}
    write_json(out / 'status.json', state)
    try:
        prep = preflight_only(args, lab, root, out, run_name)
    except Exception as exc:
        state.update(status='failed_preflight', error=repr(exc), finished_utc=utc(), fusion_started=False)
        write_json(out/'status.json',state)
        (out/'failure-traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        print(json.dumps({'status':state['status'],'output':str(out),'error':state['error'],'ROS_nodes_started':False}),flush=True)
        return 1
    state['preflight'] = prep
    if not args.execute:
        state.update(status='read_only_preflight_passed', finished_utc=utc(), fusion_started=False)
        write_json(out/'status.json', state)
        print(json.dumps({'status':state['status'],'output':str(out),'checks':prep['checks'],'ROS_nodes_started':False}),flush=True)
        return 0
    processes, observer, sensor_bag, package = None, None, None, None
    expected, before_log, player, launcher, core = {}, {}, None, None, None
    error, interrupted, bag_completed = None, False, False
    try:
        sensor_bag = Path('/opt/gs-slam-lab/fast-livo2-data/hilti2022-recovery-003-exp14/rosbags/exp14_basement_2.bag')
        if sensor_bag.stat().st_size != EXPECTED_BYTES or digest(sensor_bag) != EXPECTED_SHA:
            raise RuntimeError('Hilti exp14 complete official size/SHA256 preflight failed')
        state['bag'] = {'path': str(sensor_bag), 'bytes': EXPECTED_BYTES, 'sha256_before': EXPECTED_SHA}
        env = ros_environment(root)
        env.update(ROS_MASTER_URI=state['private_master_uri'], ROS_IP='127.0.0.1',
            ROS_LOG_DIR=str(out / 'ros-node-logs'), ROS_HOME=str(out / 'ros-home'),
            OMP_NUM_THREADS='4')
        env.pop('ROS_HOSTNAME', None)
        os.environ.update(env)
        for path in reversed(env.get('PYTHONPATH', '').split(':')):
            if path:
                sys.path.insert(0, path)
        package = Path(subprocess.check_output(['rospack','find','fast_livo'], env=env, timeout=30).decode().strip()).resolve()
        if package != root / 'catkin_ws/src/FAST-LIVO2':
            raise RuntimeError('rospack resolved an unexpected fast_livo package: ' + str(package))
        binary = root / 'catkin_ws/devel/lib/fast_livo/fastlivo_mapping'
        if not binary.is_file() or not os.access(str(binary), os.X_OK):
            raise RuntimeError('Compiled fastlivo_mapping executable is unavailable')
        write_json(out / 'source-before.json', pinned_files(package,
            lab / 'outputs/fast-livo2-source-audit-001/repository-acquisition.json', args.source_override_receipt))
        state['binary'] = {'path': str(binary), 'sha256': digest(binary)}
        state['runtime_environment'] = {k: env.get(k) for k in
            ['ROS_DISTRO','ROS_MASTER_URI','ROS_IP','ROS_LOG_DIR','ROS_HOME','OMP_NUM_THREADS','ROS_PACKAGE_PATH']}
        software = {'python': sys.version, 'platform': platform.platform()}
        for label, command in [('ros',['rosversion','-d']), ('opencv_pcl',['pkg-config','--modversion','opencv4','pcl_common-1.10']), ('binary_ldd',['ldd',str(binary)]),
                ('ros_package_versions',['dpkg-query','-W','ros-noetic-roscpp','ros-noetic-rosbag','ros-noetic-image-transport','ros-noetic-compressed-image-transport'])]:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=30)
            software[label] = {'command': command, 'exit_code': result.returncode,
                'output': result.stdout.decode(errors='replace')}
        write_json(out / 'software.json', software)
        lock_file = lab / 'scripts/fast_livo2_linux/dependency-lock.json'
        if lock_file.is_file():
            shutil.copy2(str(lock_file), str(out / 'dependency-lock.json'))
        import rosbag
        with rosbag.Bag(str(sensor_bag), 'r') as bag:
            info = bag.get_type_and_topic_info()
            expected = {topic: int(info.topics[topic].message_count) for topic in ['/hesai/pandar','/alphasense/imu','/alphasense/cam0/image_raw']}
            write_json(out / 'bag-index.json', {'topics': {topic: {'msg_type': row.msg_type,
                'message_count': int(row.message_count)} for topic, row in info.topics.items()},
                'start_ros_record_time_s': bag.get_start_time(), 'end_ros_record_time_s': bag.get_end_time(),
                'all_expected_message_count': bag.get_message_count(),
                'scope': 'Complete bag index metadata, not proof of estimator processing.'})
        # Never attach to or shut down another ROS master. A private port must be free.
        with socket.socket() as port_probe:
            port_probe.bind(('127.0.0.1', args.port))
        if master_ready(state['private_master_uri']):
            raise RuntimeError('Private master port became occupied; refusing to attach')
        before_log = inventory(package / 'Log')
        if sum(row['bytes'] for row in before_log.values()) > 2 * 1024**3:
            raise RuntimeError('Existing Log snapshot exceeds 2 GiB; preserve and review storage before replay')
        shutil.copytree(str(package / 'Log'), str(out / 'official-log-before'), copy_function=shutil.copy2)
        write_json(out / 'official-log-before-inventory.json', before_log)
        for folder in ['Log/image','Log/result','Log/pcd']:
            (package / folder).mkdir(parents=True, exist_ok=True)
        wrapper = out / 'actual-run.launch'
        shutil.copy2(str(out/'candidate-run.launch'),str(wrapper))
        planned = subprocess.run(['roslaunch','--dump-params',str(wrapper)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        (out / 'planned-parameters.yaml').write_bytes(planned.stdout)
        if planned.returncode:
            raise RuntimeError('ROS launch parameter parse failed')
        processes = OwnedProcesses(out, env)
        core = processes.start('roscore', ['roscore','-p',str(args.port)])
        deadline = time.monotonic() + 30
        while not master_ready(state['private_master_uri']):
            if core.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Owned roscore did not become ready')
            time.sleep(0.2)
        observer = Monitor(out)
        launcher = processes.start('roslaunch', ['roslaunch','--wait',str(wrapper)])
        deadline = time.monotonic() + 45
        while not launch_ready(state['private_master_uri']):
            if launcher.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Estimator raw-image/LiDAR/IMU subscriptions did not become ready')
            time.sleep(0.5)
        actual = subprocess.run(['rosparam','dump',str(out / 'actual-parameters.yaml')],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        if actual.returncode:
            raise RuntimeError('Failed to save actual ROS parameters')
        desired = {'/use_sim_time': True, '/common/img_en': int(args.mode == 'livo'),
            '/evo/seq_name': run_name, '/evo/pose_output_en': True,
            '/pcd_save/pcd_save_en': False, '/pcd_save/colmap_output_en': False,
            '/image_save/img_save_en': args.mode == 'livo', '/image_save/interval': args.image_interval,
            '/preprocess/lidar_type':5, '/preprocess/hilti_en':True, '/common/lid_topic':'/hesai/pandar',
            '/common/imu_topic':'/alphasense/imu','/common/img_topic':'/alphasense/cam0/image_raw',
            '/time_offset/img_time_offset':0.0,'/time_offset/imu_time_offset':0.0, '/time_offset/lidar_time_offset':0.0}
        parameter_checks = {}
        for name, wanted in desired.items():
            response = master(state['private_master_uri']).getParam('/gs_replay_preflight', name)
            actual_value = response[2] if response[0] == 1 else None
            parameter_checks[name] = {'expected': wanted, 'actual': actual_value,
                'passed': response[0] == 1 and actual_value == wanted}
        write_json(out / 'actual-parameter-checks.json', parameter_checks)
        if not all(row['passed'] for row in parameter_checks.values()):
            raise RuntimeError('Actual private-master parameters do not match run controls')
        node_response = master(state['private_master_uri']).lookupNode('/gs_replay_preflight', '/laserMapping')
        state['fusion_node_pid'] = master(node_response[2]).getPid('/gs_replay_preflight')[2]
        state['fusion_child_exit_code'] = None
        state['fusion_exit_scope'] = 'Actual terminal roslaunch child records are required after cleanup; parent exit never substitutes.'
        state.update(status='actual_replay_running', fusion_started=True)
        write_json(out / 'status.json', state)
        player = processes.start('rosbag-play', ['rosbag','play','--clock','--delay=3',
            '--rate=0.5',str(sensor_bag),'--topics','/hesai/pandar','/alphasense/imu','/alphasense/cam0/image_raw'])
        playback_start, progress_at = time.monotonic(), time.monotonic()
        while player.poll() is None:
            if launcher.poll() is not None or core.poll() is not None or not mapper_alive(state['private_master_uri']):
                raise RuntimeError('Owned fusion node/master exited during playback')
            if time.monotonic() - playback_start > 600:
                raise RuntimeError('Playback exceeded the 600 s functional-run deadline')
            if time.monotonic() - progress_at >= 15:
                print('Actual replay progress mode=%s wall_s=%.1f output=%s' %
                    (args.mode, time.monotonic()-playback_start, out), flush=True)
                progress_at = time.monotonic()
            time.sleep(0.5)
        if player.returncode:
            raise RuntimeError('rosbag play returned ' + str(player.returncode))
        bag_completed = True
        state['bag_playback_completed_utc'] = utc()
        drain_start = time.monotonic()
        while time.monotonic() - drain_start < 60:
            if not mapper_alive(state['private_master_uri']):
                raise RuntimeError('Fusion node exited during post-playback drain')
            elapsed = time.monotonic()-drain_start
            if elapsed >= 15 and time.monotonic()-observer.last_output_wall >= 5:
                break
            time.sleep(0.5)
        state['drain'] = {'wall_s': time.monotonic()-drain_start,
            'no_output_activity_for_s': time.monotonic()-observer.last_output_wall,
            'scope': 'Waited >=15 s and for >=5 s output inactivity, up to60 s. Buffer emptiness/absence of drops is not observable.'}
        processes.stop(launcher)
    except KeyboardInterrupt:
        interrupted, error = True, 'Caller interrupted this owned run'
    except Exception as exc:
        error = repr(exc)
        (out / 'failure-traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        cleanup_errors = []
        if processes:
            # Player, then mapper launch. Keep the private master alive until observer snapshots.
            for child in [player, launcher]:
                if child is not None:
                    try:
                        processes.stop(child)
                    except Exception as exc:
                        cleanup_errors.append('owned stop: ' + repr(exc))
        observed = None
        if observer:
            try:
                observed = observer.save(expected)
            except Exception as exc:
                cleanup_errors.append('observer save: ' + repr(exc))
        if processes:
            try:
                state['processes'] = processes.finish()
                state['owned_process_cleanup_actions'] = processes.actions
            except Exception as exc:
                cleanup_errors.append('owned process finalization: ' + repr(exc))
            if master_ready(state['private_master_uri']):
                cleanup_errors.append('Private master still responds after owned cleanup; no unrelated process was killed')
        try:
            if package and before_log:
                after_log = inventory(package / 'Log')
                write_json(out / 'official-log-after-inventory.json', after_log)
                changed = {name: row for name, row in after_log.items() if before_log.get(name) != row}
                for name in changed:
                    dest = out / 'official-log-after' / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(package / 'Log' / name), str(dest))
                state['changed_log_files'] = changed
                state['removed_log_files'] = sorted(set(before_log)-set(after_log))
                state['pre_vio_trajectory'] = trajectory_summary(package / 'Log/result' / (run_name+'.txt'),
                    'IMU body, sensor last_lio_update_time; after LiDAR update, before same-cycle VIO update')
                state['sampled_post_vio_trajectory'] = trajectory_summary(package / 'Log/image/image_poses.txt',
                    'IMU body, sensor vio_time; sampled after VIO at image_save/interval') if args.mode == 'livo' else {'enabled': False}
                write_json(out / 'source-after.json', pinned_files(package,
                    lab / 'outputs/fast-livo2-source-audit-001/repository-acquisition.json', args.source_override_receipt))
                state['pinned_source_files_unchanged_after_run'] = True
            if sensor_bag and sensor_bag.is_file():
                state.setdefault('bag', {})['sha256_after'] = digest(sensor_bag)
                state['bag']['unchanged'] = state['bag']['sha256_after'] == EXPECTED_SHA
        except Exception as exc:
            cleanup_errors.append('output/source archival: ' + repr(exc))
        if cleanup_errors:
            state['cleanup_errors'] = cleanup_errors
            error = error or 'Output evidence/cleanup did not complete'
        log = out / 'roslaunch.log'
        if log.exists():
            state['visual_log_evidence'] = visual_log_summary(log)
            write_json(out / 'visual-log-evidence.json', state['visual_log_evidence'])
            deaths = [line for line in log.read_text(errors='replace').splitlines() if 'process has died' in line]
            state['reported_roslaunch_child_deaths'] = deaths
            if deaths:
                error = error or 'ROS launch reported a child process death; inspect archived actual logs'
        child_exits = child_exit_summary(out, state.get('fusion_node_pid'))
        write_json(out / 'child-exit-evidence.json', child_exits)
        state['child_exit_evidence'] = child_exits
        codes = child_exits['children']['laserMapping-1']['exit_codes']
        state['fusion_child_exit_code'] = codes[-1] if codes else None
        checks = {'required_roslaunch_children_exit_zero': child_exits['all_required_children_exit_zero'],
            'bag_playback_exit_zero': bag_completed,
            'finite_nonempty_pre_vio_trajectory': bool(state.get('pre_vio_trajectory',{}).get('rows',0)) and not state.get('pre_vio_trajectory',{}).get('invalid_rows',[]),
            'pre_vio_quaternions_unit': state.get('pre_vio_trajectory',{}).get('all_quaternions_unit_within_1e_3',False),
            'observed_registered_scan_output': bool(observed and observed['topics'].get('/cloud_registered',{}).get('count',0)),
            'observed_native_raw_image_input': bool(observed and observed['topics'].get('/alphasense/cam0/image_raw',{}).get('count',0)),
            'pinned_source_files_unchanged': state.get('pinned_source_files_unchanged_after_run',False),
            'bag_hash_unchanged': state.get('bag',{}).get('unchanged',False)}
        if observed:
            checks['observer_callbacks_without_decode_errors'] = not observed['callback_errors']
            checks['observed_odometry_finite'] = bool(observed['topics'].get('/aft_mapped_to_init',{}).get('count',0)) and not observed['topics'].get('/aft_mapped_to_init',{}).get('invalid_messages',0)
            checks['observed_registered_scan_xyz_finite'] = not observed['topics'].get('/cloud_registered',{}).get('nonfinite_xyz_points',0)
        if args.mode == 'livo':
            checks['sampled_finite_post_vio_trajectory'] = bool(state.get('sampled_post_vio_trajectory',{}).get('rows',0)) and not state.get('sampled_post_vio_trajectory',{}).get('invalid_rows',[])
            checks['post_vio_quaternions_unit'] = state.get('sampled_post_vio_trajectory',{}).get('all_quaternions_unit_within_1e_3',False)
            checks['observed_visual_image_output'] = bool(observed and observed['topics'].get('/rgb_img',{}).get('count',0))
            checks['observed_nonzero_visual_candidates'] = bool(state.get('visual_log_evidence',{}).get('markers',{}).get('retrieved_visual_candidates',{}).get('nonzero_lines',0))
        state.update(finished_utc=utc(), bag_playback_completed=bag_completed,
            runtime_evidence_checks=checks, error=error, caller_interrupted=interrupted,
            status='failed_or_incomplete' if error or not all(checks.values()) else 'actual_local_functional_replay_evidence_passed')
        write_json(out / 'status.json', state)
        print(json.dumps({'status': state['status'], 'output': str(out), 'error': error,
            'checks': checks}, indent=2), flush=True)
    return 0 if state['status'] == 'actual_local_functional_replay_evidence_passed' else 1


if __name__ == '__main__':
    sys.exit(main())
