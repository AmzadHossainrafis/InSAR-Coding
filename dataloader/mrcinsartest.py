import os
import numpy as np
import matplotlib as plt
def readShortComplex(fileName, width=1):
    """Read scomplex data
        
       Usage example:
       slc = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/rslc/20140611.rslc', width=1000)
    """
    return np.fromfile(fileName, '>i2').astype(np.float).view(np.complex).reshape(-1, width)


def readFloatComplex(fileName, width=1):
    """Read fcomplex data
        
       Usage example:
       ifg = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20131217_20131228.diff.orb_cor', width=1000)
    """
    return np.fromfile(fileName, '>c8').astype(np.complex).reshape(-1, width)


def readFloat(fileName, width=1):
    return np.fromfile(fileName, '>f4').astype(np.float).reshape(-1, width)


def writeShortComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.copy().view(np.float).astype('>i2').tofile(out_file)
    out_file.close()


def writeFloatComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>c8').tofile(out_file)
    out_file.close()


def writeFloat(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>f4').tofile(out_file)
    out_file.close()

'''ifg = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20131217_20131228.diff.orb_cor', width=1000)
print(ifg)
[[-14972. +4234.j -14917. +3884.j -15232.+13812.j ... -15544.+26986.j
   17490.-13638.j -15334.-26879.j]
 [ 16918.+29380.j -15564.-31845.j  17560.+23695.j ...  18406.-32440.j
   18644.+11885.j  18558.+27695.j]- 
 [-15188.+24215.j -14933.+32497.j -14886.+25050.j ... -15280. +6430.j
  -15437.-29174.j  17009.-21744.j]
 ...1
 [-14859.-27125.j -15082.+25781.j -15171.+23120.j ... -14547.-29578.j
   18465.+25668.j -14432.-22658.j]
 [ 18070.+13382.j -14622.+24897.j  17959.-27988.j ...  17086.+20400.j
  -15323.+12023.j -15217.+16241.j]
 [-14878.-29076.j -14910.-25661.j  17197.+17242.j ... -14539.+11328.j
  -14751.-25702.j -14722.+29069.j]]
'''
#slc = readShortComplex('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/rslc/20140611.rslc', width=1000)
#print(slc)
#print(type(slc))
'''[[  86. -40.j  -67.  +4.j  -86.  +4.j ...   13.  +8.j  -16. -56.j
   -20. -49.j]
 [  22. -63.j  -48. +11.j  -95. +58.j ...   40.  +8.j   12. -24.j
   -19. -18.j]
 [ -20. -94.j   17. -26.j  -44. +70.j ...   28. +19.j   25. -24.j
   -24. -10.j]
 ...
 [ 106. +29.j  -51.-134.j -110.-133.j ...  246. -99.j  714.-101.j
   571.-384.j]
 [ 114. -30.j  -33.-110.j  -48.-121.j ...  502. -51.j  975.+245.j
   619.-227.j]
 [ 122. +25.j   16. -73.j   46.-107.j ...  435.+192.j  770.+431.j
   270. -91.j]] <class 'numpy.ndarray'>'''

te= readFloat('/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/rslc/20140611.rslc', width=1000)
print(te)
