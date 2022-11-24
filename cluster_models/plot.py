import matplotlib.pyplot as plt
import numpy as np

rec = False
# folders = ['Drot-arrows-D4_5_6-Linfonce-ED3-N10-Mresnet-Aresnet-IDours2', 'Drot-arrows-D4_5_6-Linfonce-ED3-N10-Mresnet-Aresnet-IDours3']
folders = ['Drot-arrows-D4_5_6-Linfonce-ED3-N1-Mresnet-Aresnet-IDbase1', 'Drot-arrows-D4_5_6-Linfonce-ED3-N1-Mresnet-Aresnet-IDbase2', 'Drot-arrows-D4_5_6-Linfonce-ED3-N1-Mresnet-Aresnet-IDbase3']
fig = plt.figure()
vals = []
for fold in folders:
    if rec:
        arr = np.load('saved_models/' + fold + '/errors_rec_val.npy')
        vals.append(arr[19] / (64*64) )
    else:
        arr = np.load('saved_models/' + fold + '/errors_val.npy')
        vals.append(arr[19] )

vals = np.array(vals)
print('mean', vals.mean(), 'std', vals.std())
