import cv2
import os
import fnmatch
import numpy as np

library_path = r'G:\420_MEN\library'
source_path = r'G:\420_MEN\raw.jpg'
target_name = 'target.jpg'

small_res = 64
source_res = None   # None for no resize
pixels = small_res*small_res
thresh = 765

img_list = os.listdir(library_path)
img_list = fnmatch.filter(img_list, '*.jpg')
lib_size = len(img_list)

# ***** get rgb in lib *********#
# error = []
# data = []
# count = 0
# for img_name in img_list:
#     try:
#         img = cv2.imread(os.path.join(library_path, img_name))
#         img_small = cv2.resize(img, (small_res, small_res))
#         imgg = img_small[:,:,0]
#         imgb = img_small[:,:,1]
#         imgr = img_small[:,:,2]
#         # obtain average rgb in pic of small res
#         small_g = np.sum(imgg, dtype=np.int32)/pixels
#         small_b = np.sum(imgb, dtype=np.int32)/pixels
#         small_r = np.sum(imgr, dtype=np.int32)/pixels
#         print('img %d is r: %d, g: %d, b: %d'% (count, small_r, small_g, small_b))
#         count = count +1
#         data.append([small_r,small_g,small_b])
#     except:
#         error.append(img_name)
# print('error:',error)
# np.save('rgbdata_64',data)




# ************** processing **************#
data = np.load('rgbdata_64.npy')
effec_size = len(data)
print(effec_size)
source = cv2.imread(source_path)
if source_res != None:
    source = cv2.resize(source,(source_res,source_res))
target = source.copy()
height, width,c = source.shape
print('file: %s | height: %d | width: %d | small_res: %d' % (source_path, height,width,small_res))
for i in range(0,width,small_res):
    for j in range(0,height,small_res):
        if (i+small_res>height or j+small_res>width):
            continue
        pacth = source[i:i+small_res, j:j+small_res, :]
        g_patch = np.sum(pacth[:,:,0])/pixels
        b_patch = np.sum(pacth[:,:,1])/pixels
        r_patch = np.sum(pacth[:,:,2])/pixels
        mini = -1
        mini_dist = thresh
        for index in range(len(data)):
            dist = abs(r_patch - data[index][0])+abs(g_patch - data[index][1])+abs(b_patch - data[index][2])
            if dist < mini_dist:
                mini = index
                mini_dist = dist
        # replace
        if mini != -1:
            pic_to_place = cv2.imread(os.path.join(library_path,img_list[mini]))
            pic_to_place = cv2.resize(pic_to_place, (small_res,small_res))
            target[i:i+small_res, j:j+small_res,:] = pic_to_place
    print('\rline %d / %d'% (i, height), end=' ')

cv2.imwrite(target_name,target)
print('Target saved in %s'% target_name)




