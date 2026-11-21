"""Same temporal audit, explicit three-candidate surface comparison layout."""
import audit_street_temporal as audit

audit.OUT=audit.LAB/'outputs/dynamic-human-street-surface-temporal-019'
audit.BRANCHES=('pose_source','appearance_only','surface_and_appearance')
audit.TITLES=('原始观测','姿态优化后的起点','仅颜色拟合','表面与颜色联合拟合')
audit.FOOTNOTE='后两组各100次全区域更新；同倍率，未锐化；10 Hz；数值来自原PNG'
audit.PANEL_MAX_WIDTH=500

if __name__=='__main__':audit.main()
