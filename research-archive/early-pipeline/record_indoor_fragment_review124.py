"""Record actual frozen render review separately from hash-pinned measurements."""
import audit_gaussianavatars_params123 as a
from datetime import datetime,timezone


def main():
    root=a.REC/'cross-subject-sh0-124';now=datetime.now(timezone.utc).isoformat()
    entries=[('indoor-directional-support124','four-orbits-support.png',
        '已查看全部四方向RGB和贡献图。放大时序版训练颜色更完整，但背部、肩袖和头颈仍有破碎。实际有效视角贡献中位数约2.36，不能归结为所有点只见过一个视角；只有约4.9%的已知前景贡献来自单一视角占比超过0.9的点。极薄高斯和偏离绑定三角形存在，但这还不是错误几何的独立真值。'),
      ('indoor-dark-fragments124','four-orbits-rgb-alpha-dark.png',
        '已实际查看全部四方向RGB、alpha和不透明黑色标记，另放大检查第27方向。新模型肩背和颈侧的部分黑条对应不透明覆盖，并非纯透明空洞。头发和深色短裤本来可为暗色，不能按暗色阈值删点。仍有真实轮廓破碎和漂浮；遮挡腿脚不验收。')]
    for name,picture,finding in entries:
        folder=root/name;result=a.read(folder/'result.json')
        assert result['state']=='readonly_completed' and result['source_hashes_unchanged']
        a.save(folder/'visual-review.json',dict(updated=now,actual_visual_review=True,image=str(folder/picture),
            image_sha256=a.sha(folder/picture),finding_zh=finding,quality_accepted=False,
            next='One1000step paired stochastic-background control; no geometry/pose changes,threshold scan or automaticlongrun.'))
    print('INDOOR_FRAGMENT_REVIEW_RECORDED')


if __name__=='__main__':main()
