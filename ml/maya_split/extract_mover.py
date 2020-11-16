import maya.cmds as mc
import csv

"""
This script can extract the mover information to a csv file.
More exactly, it will extract the range of a mover to a csv file.
For now, I only consider the transformation and translate information
"""

movers = [u'Mery_ac_rg_brow01', u'Mery_ac_rg_brow02', u'Mery_ac_rg_brow03', u'Mery_ac_lf_brow01',
          u'Mery_ac_lf_brow02', u'Mery_ac_lf_brow03', u'Mery_ac_general_lf_brow',
          u'Mery_ac_rg_entrecejo', u'Mery_ac_lf_entrecejo',
          u'Mery_ac_rg_nose', u'Mery_ac_lf_nose',
          u'Mery_ac_rg_corner_control', u'Mery_ac_lf_corner_control',
          u'Mery_ac_cn_jaw_control', u'Mery_ac_jaw_front', u'Mery_ac_rg_moflete',
          u'Mery_ac_lf_moflete', u'Mery_ac_up_rg_lip_inout', u'Mery_ac_up_lf_lip_inout',
          u'Mery_ac_dw_rg_lip_inout', u'Mery_ac_dw_lf_lip_inout', u'Mery_ac_cn_mouth_move',
          u'Mery_ac_rg_cheekbone', u'Mery_ac_lf_cheekbone', u'Mery_ac_cn_inout_mouth',
          u'Mery_ac_upLip_01_control', u'Mery_ac_upLip_02_control',
          u'Mery_ac_upLip_03_control', u'Mery_ac_upLip_04_control',
          u'Mery_ac_upLip_05_control', u'Mery_ac_loLip_05_control',
          u'Mery_ac_loLip_04_control', u'Mery_ac_loLip_03_control',
          u'Mery_ac_loLip_02_control', u'Mery_ac_loLip_01_control',
          u'Mery_ac_rg_tinyCorner_control', u'Mery_ac_lf_tinyCorner_control',
          u'Mery_ac_lf_stickyLips_control', u'Mery_ac_rg_stickyLips_control',
          u'Mery_ac_rg_inffLid_move_sg', u'Mery_ac_rg_supfLid_move_sg',
          u'Mery_ac_lf_supfLid_move_sg', u'Mery_ac_lf_infLid_move_sg',
          u'Mery_ac_rg_pupila', u'Mery_ac_rg_iris', u'Mery_ac_lf_iris',
          u'Mery_ac_lf_pupila', u'Mery_ac_general_rg_brow',
          u'Mery_ac_lf_suplid01_sg', u'Mery_ac_lf_suplid02_sg',
          u'Mery_ac_lf_suplid03_sg', u'Mery_ac_lf_lowlid01_sg',
          u'Mery_ac_lf_lowlid02_sg', u'Mery_ac_lf_lowlid03_sg',
          u'Mery_ac_rg_suplid01_sg', u'Mery_ac_rg_suplid02_sg',
          u'Mery_ac_rg_suplid03_sg', u'Mery_ac_rg_lowlid01_sg',
          u'Mery_ac_rg_lowlid02_sg', u'Mery_ac_rg_lowlid03_sg']

attributes = ['translateX', 'translateY', 'translateZ',
              'rotateX', 'rotateY', 'rotateZ',
              'scaleX', 'scaleY', 'scaleZ']

# short name for attributes
attr_sn = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']
