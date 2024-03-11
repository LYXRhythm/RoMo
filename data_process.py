import cv2
import h5py
import numpy as np

# from mayavi import mlab

# # https://developer.aliyun.com/article/1078480
# def project_2d(points, flat, pic_savepath):
#     A, B, C, D = flat
#     distance = A**2 + B**2 + C**2
#     t = -(A*points[:, 0] + B*points[:, 1] + C*points[:, 2] + D)/distance
#     x = A*t + points[:, 0]
#     y = B*t + points[:, 1]
#     z = C*t + points[:, 2]
#     project_point = np.array([x, y, z]).T
    
#     mlab.figure(bgcolor=(0.95, 0.95, 0.85), size=(640, 640))
#     mlab.points3d(x, y, z, y, mode='point', colormap='copper')#, color=(0.15, 0.15, 1))

#     mlab.options.offscreen=False
#     mlab.savefig(filename=pic_savepath, size=(30, 30))
#     mlab.close()

def main(ds_path):
    ## train
    # imgs_path = "./dataset/3dmnist/imgs/train/"
    # points_path = "./dataset/3dmnist/points/train/"
    # img_txt = "./imgs_train_list.txt"
    # point_txt = "./points_train_list.txt"

    ## test
    imgs_path = "./dataset/3dmnist/imgs/test/"
    points_path = "./dataset/3dmnist/points/test/"
    points3d_path = "./dataset/3dmnist/points3d/test/"
    img_txt = "./imgs_test_list.txt"
    point_txt = "./points_test_list.txt"
    point3d_txt = "./points3d_test_list.txt"

    img_f = open(img_txt, 'w', encoding = 'utf-8')
    point_f = open(point_txt, 'w', encoding = 'utf-8')
    point3d_f = open(point3d_txt, 'w', encoding = 'utf-8')
    count = 0
    with h5py.File(ds_path, "r") as hf:    
        for ii in hf.keys():
            data = hf[ii]
            datalist = (data["img"][:], data["points"][:], data.attrs["label"])

            img_path = imgs_path + ii + ".png" + " " + str(datalist[2]) + "\n"
            point_path = points_path + ii + ".png" + " " + str(datalist[2]) + "\n"
            point3d_path = points3d_path + ii + ".npy" + " " + str(datalist[2]) + "\n"

            img_f.write(img_path)
            # point_f.write(point_path)
            point3d_f.write(point3d_path)

            ## img
            # img = np.array(datalist[0])*255
            # cv2.imwrite(img_path.split(" ")[0], img)

            # ## point
            # project_2d(datalist[1], (1, 0.65, 0, 0), point_path.split(" ")[0])

            # ## point3d
            np.save(point3d_path.split(" ")[0], datalist[1])

            print(count, " / ", len(hf.keys()))
            count = count + 1

    img_f.close()
    point_f.close()

if __name__ == "__main__":
    ori_dataset = "./dataset/3dmnist/origin/test_point_clouds.h5"
    main(ori_dataset)