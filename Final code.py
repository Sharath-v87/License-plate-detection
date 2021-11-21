import os
import object_detection
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import easyocr
import string
import tkinter as tk
from tkinter import ttk,Label,Canvas,Text
from tkinter import filedialog as fd
from PIL import Image, ImageTk

reader = easyocr.Reader(['en'])


CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

def pred(path):
    img = cv2.imread(path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    return image_np_with_detections,detections

detection_threshold = 0.9
def ocre(image,detections):
    scores = list(filter(lambda x : x > detection_threshold,detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]
    
    for idx,box in enumerate(boxes):
        roi = box*[height,width,height,width]
        thres = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        temp=thres
        sup_op = super_res(temp)
        thres = cv2.cvtColor(thres, cv2.COLOR_BGR2GRAY)
        thres = thres.astype(np.uint8)
        region = cv2.threshold(thres, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #plt.imshow(region)
        ocr_res = reader.readtext(temp,allowlist=string.ascii_uppercase + "0123456789")
        return ocr_res

def super_res(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "LapSRN_x8.pb"
    sr.readModel(path)
    sr.setModel("lapsrn",8)
    result = sr.upsample(img)
    return result

def indian_rule(ocr_res):
    numtolet={0:'O',4:'A',1:'I',5:'S',8:'B',7:'V',3:'S'}
    lettonum={'O':0,'L':4,'I':1,'S':5,'B':8,'A':4,'J':3,'T':1,'Z':2,'G':6,'Q':0,'D':0,'F':'F'}
    state_code=['AN','AP','AR','AS','BR','CH','CG','DL','GJ','HR','JK','JH','KA','KL','MP','MH','MN','ML','MZ','OD','PB','RJ','SK','TN','TS','TR','UP','UK','WB']
    sec_h={'H':'M'}
    fir_m={'A':'H','M':'H'}
    sec_t={'M':'N',5:'S','H':'N'}
    sec_b={'H':'W'}
    alpha=string.ascii_uppercase
    numb = '0123456789'
    test=''
    for j in range(0,len(ocr_res)):
        if len(ocr_res)<1:
            continue
        if len(ocr_res)>1:
            test=test+ocr_res[j]
            if len(test) > 10:
                if test[:3] == 'IND':
                    test = test[3:]
                elif test[:2] == 'ND':
                    test = test[2:]
        else:
            test=ocr_res[j]
    l=list(test)
    if len(l)>10:
        if l[0] == 'I' or l[0] == 'F':
            l=l[1:]
        else:
            l.pop()

    try:
        if l[0] in alpha and l[1] in alpha:
            pass
        elif not(l[1] in alpha) and l[0] in alpha:
            l[1]=numtolet[int(l[1])]
        elif not(l[0] in alpha) and l[1] in alpha :
            l[0]=numtolet[int(l[0])]
        else:
            l[1]=numtolet[int(l[1])]
            l[0]=numtolet[int(l[0])]


        if l[2] in numb and l[3] in numb:
            pass
        elif not(l[3] in numb) and l[2] in numb :
            l[3] = lettonum[l[3]]
        elif not(l[2] in numb) and l[3] in numb:
            l[2]=lettonum[l[2]]
        else:
            l[2]=lettonum[l[2]]
            l[3] = lettonum[l[3]]

        if l[4] in alpha and l[5] in alpha:
            pass
        elif not(l[5] in alpha) and l[4] in alpha :
            l[5]=numtolet[int(l[5])]
        elif not(l[4] in alpha) and l[5] in alpha:
            l[4]=numtolet[int(l[4])]
        else:
            l[4]=numtolet[int(l[4])]
            l[5]=numtolet[int(l[5])]

        if l[6] in numb and l[7] in numb and l[8] in numb and l[9] in numb :
            pass
        if not(l[6] in numb):
            l[6] = lettonum[l[6]]
        if not(l[7] in numb):
            l[7] = lettonum[l[7]]
        if not(l[8] in numb):
            l[8] = lettonum[l[8]]
        if not(l[9] in numb):
            l[9] = lettonum[l[9]]

    except:
        pass
    if len(l)>10:
        l=l[0:10]
    state = str("".join(str(i) for i in l[0:2]))
    if state in state_code:
        pass
    else:
        flag = 0
        if l[1] == 'H':
            a = l[0]
            b = sec_h[a]
            l[0]= b
            flag = 1
        if l[0] == 'M'and flag == 0:
            l[1] = fir_m[l[1]]
        if l[1] == 'A':
            l[0] = 'K'
        if l[0] == 'T':
            l[1] = sec_t[l[1]]
        if l[0] == 'R':
            l[1] = 'J'
        if l[1] == 'G':
            l[0] = 'C'
    fin = str("".join(str(i) for i in l))
    return fin

root = tk.Tk()
root.title('LPR')
root.resizable(700,700)
root['background']="#4c0099"
root.geometry('700x700')

def selected_file():
    global text
    filetype=(('jpg files','*.jpg'),('png files','*.png'))
    filename = fd.askopenfilename(title = 'select the file',filetypes = filetype)
    img,flag= pred(filename)
    ocr_res = ocre(img,flag)
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    top = tk.Toplevel()
    top.title('output')
    top.geometry('700x700')
    top.resizable(700,700)
    lab = tk.Label(top,text='Recognition completed!').pack()
    imge = Label(top, image=imgtk)
    imge.image = imgtk
    imge.place(x=150, y=200)
    text = Text(top, height=8)
    fin_text=''
    for i in range(0,len(ocr_res)):
        fin_text+=ocr_res[i][1]
    text = tk.Label(top, text='OCR output : ' + fin_text + '\n' + 'After applying pattern rule : '+ indian_rule(fin_text) )
    text.place(x=0,y=90)
    text.pack()
    close_button = ttk.Button(top,text='Go back',command=top.destroy)
    close_button.place(x=305,y=600).pack()

def live():
    vide = cv2.VideoCapture(0)
    sec = tk.Toplevel()
    sec.title('live output')
    sec.geometry('700x700')
    sec.resizable(700,700)
    
    flag, image = vide.read()
    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    im = Image.fromarray(image_np_with_detections)
    resized_image= im.resize((300,300), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=resized_image)
    imge = Label(sec, image=imgtk)
    imge.image = imgtk
    imge.place(x=150, y=200)

    
    detection_threshold = 0.9
    
    image = image_np_with_detections
    scores = list(filter(lambda x : x > detection_threshold,detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]
    
    for idx,box in enumerate(boxes):
        temp=thres
        sup_op = super_res(temp)
        thres = cv2.cvtColor(thres, cv2.COLOR_BGR2GRAY)
        thres = thres.astype(np.uint8)
        region = cv2.threshold(thres, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #plt.imshow(region)
        ocr_res = reader.readtext(temp,allowlist=string.ascii_uppercase + "0123456789")
        fin_text=''
        for i in range(0,len(ocr_res)):
            fin_text+=ocr_res[i][1]
        text = tk.Label(sec, text='OCR output : ' + fin_text + '\n' + 'After applying pattern rule : '+ indian_rule(fin_text) )
        text.place(x=0,y=90)
        text.pack()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        sec.destroy()

    
open_button = ttk.Button(root,text='Select the image',command=selected_file)

open_button.pack(expand=True)

Live_button = ttk.Button(root,text='live detection',command=live)
Live_button.pack(expand=True)


root.mainloop()