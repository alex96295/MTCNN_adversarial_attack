from models import mtcnn
import cv2
import os
import json
import torch
import numpy as np

mtcnn_net = mtcnn.MTCNN(keep_all=True, device=None)

#image = cv2.cvtColor(cv2.imread("multiface.jpg"), cv2.COLOR_BGR2RGB)

img_dir = "C:/Users/Alessandro/Desktop/FDDB_dataset/FDDB-folds-anno/"
img_init_path = "C:/Users/Alessandro/Desktop/FDDB_dataset/FDDB_image_dataset/"
destination_path = "C:/Users/Alessandro/PycharmProjects/mtcnn_pytorch_adversarial/FDDB_test_results/clean_results/"

n = 0
#Loop over cleaned images
for img_subset in os.listdir(img_dir):
    n+=1
    print('SUBSET NO. ' + str(n))
    img_subset_file_path = os.path.join(img_dir, img_subset)
    textfile_images = open(img_subset_file_path, 'r')
    k = 0
    for single_img in textfile_images:
        k+=1
        print('IMAGE NO. ' + str(k) + ' OF SUBSET NO. ' + str(n))
        single_img = single_img.strip('\n')
        #print(single_img)
        single_img_name = single_img + ".jpg"
        print(single_img_name)
        single_image_path = os.path.join(img_init_path, single_img_name)
        #print(single_image_path)

        destination_name_label = single_img_name.replace('.jpg', '.txt')
        destination_name_label = destination_name_label.replace('/', '_')
        destination_label = os.path.join(destination_path, 'mtcnn_labels/', destination_name_label)
        destination_name_images = single_img_name.replace('/', '_')

        destination_image = os.path.join(destination_path, 'mtcnn_images_set/', destination_name_images)

        image = cv2.cvtColor(cv2.imread(single_image_path), cv2.COLOR_BGR2RGB)

        cv2.imwrite(destination_image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        img = torch.from_numpy(image).float()

        boxes, probs, landmarks = mtcnn_net.detect(img, landmarks=True)

        results = []

        destination_image_detected = os.path.join(destination_path, 'mtcnn_detected_images/', destination_name_images)

        textfile = open(destination_label, 'w+')
        if boxes is not None:
            for box, conf, landmark in zip(boxes, probs, landmarks):

                x_left_box = max(0, box[0])
                y_top_box = max(0, box[1])
                w_box = box[2] - box[0]
                h_box = box[3] - box[1]
                results.append({
                    'box': [max(0, int(box[0])), max(0, int(box[1])),
                            int(box[2] - box[0]), int(box[3] - box[1])],
                    'confidence': conf,
                    'keypoints': {
                        'left_eye': (int(landmark[0][0]), int(landmark[0][1])),
                        'right_eye': (int(landmark[1][0]), int(landmark[1][1])),
                        'nose': (int(landmark[2][0]), int(landmark[2][1])),
                        'mouth_left': (int(landmark[3][0]), int(landmark[3][1])),
                        'mouth_right': (int(landmark[4][0]), int(landmark[4][1])),
                    }
                }
                )

                textfile.write(f'{0} {round(float(x_left_box),3)} {round(float(y_top_box),3)} {round(float(w_box),3)} {round(float(h_box),3)} {round(float(landmark[0][0]),3)} {round(float(landmark[0][1]),3)} {round(float(landmark[1][0]),3)} {round(float(landmark[1][1]),3)} {round(float(landmark[2][0]),3)} {round(float(landmark[2][1]),3)} {round(float(landmark[3][0]),3)} {round(float(landmark[3][1]),3)} {round(float(landmark[4][0]),3)} {round(float(landmark[4][1]),3)} {round(float(conf),6)}\n')

            print(results)
            textfile.close()
            with open('./multiface_res.json', 'w') as fp:
               json.dump(str(results), fp)

            for i in range(len(results)):
                bounding_box = results[i]['box']
                keypoints = results[i]['keypoints']
                cv2.rectangle(image,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0,155,255), 2)

                cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

        cv2.imwrite(destination_image_detected, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


