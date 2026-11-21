"""Audit cached source/control and new joint024 temporal comparisons."""
import audit_street_temporal as audit

audit.OUT=audit.LAB/'outputs/dynamic-human-street-joint-pose-surface-024/temporal'
audit.BRANCHES=('pose_source','surface_and_appearance','joint_pose_surface')
audit.TITLES=('原始观测','共同起点','表面与颜色','姿态、表面与颜色')
audit.FOOTNOTE='后两组各100次全区域更新；同倍率，未锐化；10 Hz；数值来自原PNG'
audit.PANEL_MAX_WIDTH=500

if __name__=='__main__':audit.main()
