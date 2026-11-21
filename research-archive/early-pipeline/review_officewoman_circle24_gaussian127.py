"""Reuse same-source read-only CPU review for the new24-view officewoman."""
import sys
import review_football_half12_gaussian126 as previous
from train_indoor_circle16_sh0124 import reused
from train_officewoman_circle24_sh0127 import OUT,N,NATIVE_WH
from review_officewoman_circle24_fit127 import joint_group_support,foot_review_crop


def compiled_review():
    edits=[
        ("Path('/mnt/d/00/记录/011')","Path('/mnt/d/00/记录/019')"),
        ('assert raw.size == mask.size == (2704,1520)','assert raw.size == mask.size == NATIVE_WH'),
        ("        raw = Image.open(source['image']).convert('RGB')",
         "        if type(source.get('face_region_is_observed')) is not bool:\n            raise ValueError('Unknown127source face visibility: review required; no implicit true.')\n        support = joint_group_support(source)\n        raw = Image.open(source['image']).convert('RGB')"),
        ("        crops = {'body':body,'feet':crop_box([body[0],body[1]+.72*(body[3]-body[1]),body[2],body[3]],raw.size)}",
         "        crops = {'body':body}\n        feet = foot_review_crop(dict(observed_native_keypoints=fit['observed_native_keypoints'][i:i+1]),support,raw.size)\n        if feet is not None:\n            crops['feet'] = feet"),
        ("if source.get('face_box_xyxy') is not None:",
         "if source['face_region_is_observed'] is True and source.get('face_box_xyxy') is not None:"),
        ('高斯_足球场半圈12帧126_','高斯_办公室环绕24帧127_'),
        ('hands_near_pockets_unknown=True','occluded_hand_details_unknown=True'),
        ('unseen_rear_unknown=True','unobserved_surface_details_unknown=True'),
        ('records.append(dict(frame=frame,images=images,metrics=metrics,',
         "records.append(dict(frame=frame,images=images,metrics=metrics,source_joint_group_supported=support,feet_region_has_supported_observations=('feet' in crops),"),
        ('FOOTBALL_TWELVE_GAUSSIAN_REVIEW_READY','OFFICEWOMAN24_GAUSSIAN_REVIEW_READY'),
    ]
    return reused(previous.main,edits,dict(OUT=OUT,N=N,NATIVE_WH=NATIVE_WH,joint_group_support=joint_group_support,foot_review_crop=foot_review_crop),'officewoman24-source-review')


def main(step):
    compiled_review()[0](step)


if __name__=='__main__':
    main(int(sys.argv[1]))
