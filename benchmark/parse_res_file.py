import numpy as np
import sys
import re

epochPattern = re.compile(r'^Epoch: \[(.*)\] \[')
sessName = sys.argv[1]
imStd = {}
imPsnr = {}
nepochs = 0
imEpoch = []
with open(sessName) as f:
    for line in f:
        #im = 'img' + str(i)
        m = epochPattern.match(line)
        if m:
            nepochs = m.group(1)
        for i in range(1,26):
            pattern = re.compile(r'img' + str(i)  + ' ')
#            print(pattern)
            if pattern.search(line):
                if i == 1: 
                    imEpoch.append(nepochs)
                tab = line.split(' ')
                std_ = float(tab[4])
                psnr_ = float(tab[2][:-2])
#                print(i, std_, psnr_)
                if i in imStd:
                    imStd[i].append(std_)
                    imPsnr[i].append(psnr_)
                else:
                    imStd[i] = [std_]
                    imPsnr[i] = [psnr_]
#print(imEpoch)
noise = {0.0: [1, 6, 11, 16, 21],
         1.0: [2, 7, 12, 17, 22],
         1.5: [3, 8, 13, 18, 23],
         2.0: [4, 9, 14, 19, 24],
         2.5: [5, 10, 15, 20, 25]}

stdt = np.array([imStd[i] for i in range(1,25)]).mean(axis=0)
psnrt = np.array([imPsnr[i] for i in range(1, 26)]).mean(axis=0)
minEpoch = np.argmin(stdt)
print('NumEpochs:', imEpoch[-1])

print('BestEpoch ', imEpoch[minEpoch])
x_avg = 'avg %.3f %.3f' %(stdt[minEpoch], psnrt[minEpoch])
print(x_avg.replace('.',','))

for n in noise:
    stdn = np.array([imStd[i] for i in noise[n]]).mean(axis=0)
    psnrn = np.array([imPsnr[i] for i in noise[n]]).mean(axis=0)
    #print(stdn.shape)
    #print(stdn)
    #minEpoch = np.argmin(stdn)
    x = '%.1f %.3f %.3f' %(n, stdn[minEpoch], psnrn[minEpoch])
    print(x.replace('.',','))
