"""Reuse the same native crops and geometry review for the ADC-only control."""
import argparse
import review_shared_offset_lr123 as review

review.ROOT=review.O/'shared-vertex-offset-adc-1000'
review.OUT=review.ROOT/'comparison'
review.SNAP=review.ROOT/'gaussian-run/step-001000'
review.BASELINE_NAME='ADC10000-2000'
review.CANDIDATE_NAME='ADC300-60'
review.RECORD_PREFIX='20261030_增密时序对照1000步_左原图_中10000-2000_右300-60'
review.NOVEL_RECORD='20261030_增密时序300-60_1000步_连续未知视角000至035.png'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--novel',action='store_true');args=parser.parse_args()
    review.novel() if args.novel else review.compare()
