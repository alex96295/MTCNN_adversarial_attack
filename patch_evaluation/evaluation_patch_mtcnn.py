#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import brambox as bb
import brambox.io.parser.box as bbb
import matplotlib.pyplot as plt
from IPython.display import display


# In[5]:


# todo NB ANNOTATIONS COINCIDES WITH CLEAN DETECTIONS! THEY ARE NOT THE GROUND-TRUTH OF INRIA DATASET, SO CLEAN RESULTS ARE 100% ASSUMED AS CORRECT
#annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/YOLOv2_d2k_adv_COCO_thys2019/test_results_mytrial/clean/yolo-labels/', class_label_map={0: 'person'})
annotations = bb.io.load('anno_darknet', "C:\Users\Alessandro\Desktop\GitLab\face_adversarial_pytorch\test_results_FDDB\clean_results\mtcnn_labels", class_label_map={0: 'person'})

clean_results = bb.io.load('det_coco', "./json_files/clean_results.json", class_label_map={0: 'person'})
patch_results_glasses_pnet_maxsc_maxprop = bb.io.load('det_coco', "./json_files/glasses_patch_maxscales_maxprop_pnet.json", class_label_map={0: 'person'})
patch_results_mouth_pnet_maxsc_maxprop = bb.io.load('det_coco', "./json_files/mouth_patch_maxscales_maxprop_pnet.json", class_label_map={0: 'person'})
patch_results_mouth_pnet_meansc_threshprop = bb.io.load('det_coco', "./json_files/mouth_patch_meanscales_threshprop_pnet.json", class_label_map={0: 'person'})
patch_results_glasses_allnets_meansc_maxprop = bb.io.load('det_coco', "./json_files/glasses_patch_meanscales_maxprop_sumallnets.json", class_label_map={0: 'person'})
patch_results_mouth_allnets_meansc_maxpropr = bb.io.load('det_coco', "./json_files/mouth_patch_meanscales_maxprop_sumallnets.json", class_label_map={0: 'person'})


# In[8]:


plt.figure()

clean = bb.stat.pr(clean_results, annotations, threshold=0.5)
p_g_pnet_maxsc_maxprop = bb.stat.pr(patch_results_glasses_pnet_maxsc_maxprop, annotations, threshold=0.5)
p_m_pnet_maxsc_maxprop = bb.stat.pr(patch_results_mouth_pnet_maxsc_maxprop, annotations, threshold=0.5)
p_m_pnet_meansc_threshprop = bb.stat.pr(patch_results_mouth_pnet_meansc_threshprop, annotations, threshold=0.5)
p_g_allnet_meansc_maxprop = bb.stat.pr(patch_results_glasses_allnets_meansc_maxprop, annotations, threshold=0.5)
p_m_allnet_meansc_maxprop = bb.stat.pr(patch_results_mouth_allnets_meansc_maxpropr, annotations, threshold=0.5)
#random_noise = bb.stat.pr(patch_results_random_noise, annotations, threshold=0.5)
#random_image = bb.stat.pr(patch_results_random_image, annotations, threshold=0.5)

#ap = bbb.ap(teddy[0], teddy[1])
#plt.plot(teddy[1], teddy[0], label=f'Teddy: mAP: {round(ap*100, 2)}%')

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

ap = bb.stat.ap(clean)
plt.plot(clean['recall'], clean['precision'], label=f'CLEAN: AP: {round(ap*100, 2)}%') #, RECALL: {round(clean["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(p_g_pnet_maxsc_maxprop)
plt.plot(p_g_pnet_maxsc_maxprop['recall'], p_g_pnet_maxsc_maxprop['precision'], label=f'GLASS_PNET_MAXSC_MAXPROP: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(p_m_pnet_maxsc_maxprop)
plt.plot(p_m_pnet_maxsc_maxprop['recall'], p_m_pnet_maxsc_maxprop['precision'], label=f'MOUTH_PNET_MAXSC_MAXPROP: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(p_m_pnet_meansc_threshprop)
plt.plot(p_m_pnet_meansc_threshprop['recall'], p_m_pnet_meansc_threshprop['precision'], label=f'MOUTH_PNET_MEANSC_THRESHPROP: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(p_g_allnet_meansc_maxprop)
plt.plot(p_g_allnet_meansc_maxprop['recall'], p_g_allnet_meansc_maxprop['precision'], label=f'GLASS_ALLNETS_MEANSC_MAXPROP: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(p_m_allnet_meansc_maxprop)
plt.plot(p_m_allnet_meansc_maxprop['recall'], p_m_allnet_meansc_maxprop['precision'], label=f'MOUTH_ALLNETS_MEANSC_MAXPROP: AP: {round(ap*100, 2)}%')

# ap = bb.stat.ap(random_noise)
# plt.plot(random_noise['recall'], random_noise['precision'], label=f'NOISE: AP: {round(ap*100, 2)}%, RECALL: {round(random_noise["recall"].iloc[-1]*100, 2)}%')
# 
# ap = bb.stat.ap(random_image)
# plt.plot(random_image['recall'], random_image['precision'], label=f'RAND_IMG: AP: {round(ap*100, 2)}%, RECALL: {round(random_image["recall"].iloc[-1]*100, 2)}%')

plt.gcf().suptitle('PR-curve MTCNN, dataset = FDDB')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig("./pr_curves/mtcnn_fddb.png")
plt.savefig("./pr_curves/mtcnn_fddb.eps")
#plt.show()

