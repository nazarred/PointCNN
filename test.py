import laspy

from data_utils import read_xyz_label_from_las_laspy

x, i, r, l, n = read_xyz_label_from_las_laspy('/home/nazar/phoenixlidar/ml/data/las_test_1/train/MiniVux_DL_PowerLines_Substation_UTM33N.las', use_hag_as_z=True)