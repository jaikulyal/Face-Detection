#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install labelme')


# In[ ]:


import tensorflow as tf
tf.config.list_physical_devices("GPU")

tf.test.is_gpu_available()


# In[3]:


import os
import time
import uuid #unique uniform identity
import cv2 


# In[10]:


IMAGES_PATH = os.path.join('data','images')
number_images = 30


# In[16]:


cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
     print('collecting images{}'.format(imgnum))
     ret, frame = cap.read()
     imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
     cv2.imwrite(imgname, frame)
     cv2.imshow('frame',frame)
     time.sleep(0.5)

     if cv2.waitKey(1) & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()


# In[3]:


import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        print("found the GPU with name",gpu)
else:
    print("failed")


# In[20]:


get_ipython().system('labelme')


# In[31]:


get_ipython().system('pip install albumentations')


# In[5]:


import tensorflow as tf 
import json
import numpy as np
from matplotlib import pyplot as plt


# In[6]:


#avoid OOm error by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[6]:


tf.keras.backend.clear_session()

def set_session(gpus: int = 0):
    num_cores = cpu_count()

    config = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
        allow_soft_placement=True,
        device_count={"CPU": 1, "GPU": gpus},
    )

    session = tf.Session(config=config)
    k.set_session(session)


# In[35]:


tf.config.list_physical_devices('GPU')

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
# In[39]:


images = tf.data.Dataset.list_files('data\\images\\*jpg',shuffle=False)


# In[5]:


images.as_numpy_iterator().next()


# In[40]:


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


# In[42]:


images = images.map(load_image)


# In[361]:


images.as_numpy_iterator().next()


# In[ ]:


type(images)


# In[ ]:


image_generator = images.batch(4).as_numpy_iterator()


# In[ ]:


plot_images = image_generator.next()


# In[43]:


fig, ax = plt.subplots(ncols=4, figsize=(10,10))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()    


# In[44]:


#manually spliting data into train and test validate


# In[45]:


90*.7 #63 to train


# In[48]:


90*.15 # 14 and 13 to test and val


# In[49]:


#move The Matching Lables
for folder in['train','test','val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):

        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data','labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath, new_filepath)
    
                           


# In[88]:


#img = cv2.imread(os.path.join('data','val','images','324ada1e-40e2-11ef-9253-f46add5887fa - Copy.jpg'))


# In[99]:


#img.shape


# In[11]:


#4.Applying Image Augmentation
#4.1 setuping Albumentations Transform Pipeline
import albumentations as alb


# In[12]:


augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='albumentations',label_fields=['class_labels']))


# In[15]:


img = cv2.imread(os.path.join('data','train','images','2ff478da-40e2-11ef-b65f-f46add5887fa.jpg'))


# In[17]:


img.shape


# In[19]:


import json
with open(os.path.join('data','train','labels','2ff478da-40e2-11ef-b65f-f46add5887fa.json'),'r') as f:
     label = json.load(f)


# In[21]:


label


# In[23]:


label['shapes'][0]['points']


# In[25]:


coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]


# In[27]:


coords


# In[29]:


img.shape


# In[31]:


import numpy as np
coords = list(np.divide(coords, [640,480,640,480]))


# In[23]:


coords


# In[33]:


augmented = augmentor(image = img, bboxes=[coords], class_labels=['face'])


# In[25]:


augmented


# In[22]:


augmented['image'].shape


# In[83]:


augmented['bboxes']
augmented['image']


# In[25]:


cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2],[450,450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:],[450,450]).astype(int)),
                       (255,0,0),2)
plt.imshow(augmented['image'])

              


# In[107]:


#building and Run Augmented pipeline
for partition in ['train','test','val']:
    for image in os.listdir(os.path.join('data',partition,'images')):
        img = cv2.imread(os.path.join('data',partition,'images',image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data',partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
                
            coords = [0,0,0,0]
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords,[640,480,640,480]))

        try:
            for x in range(60):
               augmented = augmentor(image = img, bboxes=[coords], class_labels=['face'])
               cv2.imwrite(os.path.join('aug_data',partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

               annotation = {}
               annotation['image'] = image

               if os.path.exists(label_path):
                   if len(augmented['bboxes']) == 0:
                       annotation['bbox'] = [0,0,0,0]
                       annotation['class'] = 0
                   else:
                       annotation['bbox'] = augmented['bboxes'][0]
                       annotation['class'] = 1
               else:
                   annotation['bbox'] = [0,0,0,0]
                   annotation['class'] = 0

               with open(os.path.join('aug_data', partition, 'labels',f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                   json.dump(annotation, f)

        except Exception as e:
            print(e)
               
                   


# In[35]:


#5.2 LOAD AUGMENTED IMAGES TO TENSORFLOW DATASET


# In[44]:


train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)


# In[46]:


test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)


# In[48]:


val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)


# In[50]:


train_images.as_numpy_iterator().next()


# In[52]:


# 6. Prepare Lables
#6.1 Build Label Loading Function


# In[54]:


def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']


# In[56]:


#6.2 Load Labels to Tensorflow dataset


# In[58]:


train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# In[60]:


test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# In[62]:


val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# In[64]:


train_labels.as_numpy_iterator().next()


# In[66]:


#7. Combine Labels and Images Samples


# In[68]:


len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)


# In[70]:


train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)


# In[72]:


test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)


# In[74]:


val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)


# In[76]:


#7.3 VIEW Images and Annotation


# In[78]:


train.as_numpy_iterator().next()[1]


# In[79]:


data_samples = train.as_numpy_iterator()


# In[80]:


res = data_samples.next()


# In[81]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample_image = res[0][idx].copy()
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                        (255,0,0), 2)
    ax[idx].imshow(sample_image)

plt.show()


# In[44]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

# Assuming res is a list of results where res[0] contains images and res[1] contains coordinates
# Example structure: res = [list_of_images, [list_of_image_metadata, list_of_coordinates]]

# Create a figure with 4 subplots (1 row, 4 columns) and set the figure size
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

# Loop through the first 4 images
for idx in range(4):
    # Extract a sample image and its corresponding coordinates
    sample_image = res[0][idx].copy()  # Make a writable copy of the image
    sample_coords = res[1][1][idx]

    # Draw a rectangle on the sample image using the coordinates
    # The coordinates are multiplied by [120, 120] to scale them appropriately
    # The color of the rectangle is set to blue (BGR: 255, 0, 0) and the thickness is set to 2
    cv2.rectangle(
        sample_image,
        tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),  # Top-left corner
        tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),  # Bottom-right corner
        (255, 0, 0),  # Color of the rectangle (blue)
        2  # Thickness of the rectangle
    )

    # Display the sample image with the rectangle in the subplot
    ax[idx].imshow(sample_image)  # Convert BGR to RGB for correct color display
    ax[idx].axis('off')  # Hide axis

# Show the plot
plt.show()


# In[45]:


#8. Building Deep Learmimg Using the Functional API
#8.1 import Layers and Base Network


# In[46]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers  import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16


# In[47]:


vgg = VGG16(include_top = False)


# In[71]:


vgg.summary()


# In[72]:


#8.3 Building instance of Network


# In[83]:


def build_model():
    input_layer = Input(shape=(120,120,3))

    vgg = VGG16(include_top=False)(input_layer)
    #classification model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    #regression model   Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker
    


# In[84]:


train.as_numpy_iterator().next()[1]


# In[85]:


#8.4 TEST NEURAL NETWORK


# In[86]:


facetracker = build_model()


# In[ ]:


facetracker.summary()


# In[ ]:


X, y = train.as_numpy_iterator().next()


# In[ ]:


y


# In[ ]:


X.shape


# In[ ]:


classes, coords = facetracker.predict(X)


# In[ ]:


classes, coords


# In[ ]:


#9.DEFINNING LOSSES
#9.1 Define OPTIMIZER and LR


# In[ ]:


len(train)


# In[ ]:


batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch


# In[ ]:


lr_decay


# In[ ]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)


# In[ ]:


#9.2 Create Localization loss and Classification Loss


# In[ ]:


def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))

    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]

    h_pred = yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size


# In[ ]:


classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss


# In[ ]:


#9.3 Test out Loss Metrics


# In[ ]:


localization_loss(y[1], coords)


# In[ ]:


classloss(y[0], classes)


# In[ ]:


regressloss(y[1], coords)


# In[ ]:





# In[ ]:


#10. TRAIN OUR NEURAL NETWORK
#10.1 Create custom Model


# In[ ]:


class FaceTracker(Model):
    def __init__(self, eyetracker,  **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):

        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)

            opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
            return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}

    def test_step(self, batch, **kwargs):
         X, y = batch
         
         classes, coords = self.model(X, training=False)
         
         batch_classloss = self.closs(y[0], classes)
         batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
         total_loss = batch_localizationloss+0.5*batch_classloss
         
         return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)
         


# In[181]:


model  = FaceTracker(facetracker)


# In[182]:


model.compile(opt, classloss, regressloss)


# In[184]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[185]:


hist = model.fit(train, epochs=40, validation_data=val, callbacks=[tensorboard_callback])


# In[183]:


#10.2 train
logdir='logs'


# In[186]:


fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='var loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val class loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show


# In[86]:


#11.1 make predection
test_data =test.as_numpy_iterator()


# In[187]:


test_sample = test_data.next()


# In[188]:


yhat = facetracker.predict(test_sample[0])


# In[189]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample_image = test_sample[0][idx].copy()
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.5:
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                        (255,0,0), 2)
        ax[idx].imshow(sample_image)
                      


# In[190]:


#11.2 save the Model


# In[191]:


from tensorflow.keras.models import load_model


# In[192]:


facetracker.save('facetracker.h5')


# In[193]:


facetracker = load_model('facetracker.h5')


# In[194]:


#REALTIME FACE DETECTION

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5:
        #controlls the main rectangle
        alpha = 0.5
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)),
                            (165,0,255), 2)
        #controls the lable rectange

       # cv2.rectangle(frame,
          #            tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
         #                         [0,-30])),
         #             tuple(np.add(np.multiply(sample_coords[2:], [450,450]).astype(int),
         #                         [80,-250])),
         #                   (255,0,0), -1)
        #control the text rended
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow('EyeTracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
            
            


# In[ ]:





# In[ ]:




