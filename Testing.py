import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import model_from_json
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img
from statistics import mean


WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 16
TEST_DIR = '/Users/Christos/PycharmProjects/kerasnew/Christos/EXPERIMENT/test500/'


def predict(model, img):
     """Run model prediction on image
#     Args:
#         model: keras model
#         img: PIL format image
#     Returns:
#         list of predicted labels and their probabilities
#     """
     x  = image.img_to_array(img)
     x = np.expand_dims(x, axis=0)
     x = preprocess_input(x)
     preds = model.predict(x)
     return preds[0]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load the weights
loaded_model.load_weights("LuISSSSSSito.h5")
print("Loaded model from disk")
loaded_model.summary()

path = 'TEST1_new/trav'

listt=[]

for i in range(74):
    filename = '/Users/Christos/PycharmProjects/kerasnew/Christos/EXPERIMENT/TEST_K/' + path + str(i) + '.jpg'
    img = image.load_img(filename, target_size=(224, 224))
    preds = predict(loaded_model, img)
    if preds[0] > preds[1] and (preds[0]-preds[1]) > 0.32 :
       print(preds[0], 'TRAV',)
     # listt.append(preds[0])
    else:
        print(preds[1] , 'non', filename)
        listt.append(preds[1])
print(len(listt))

print(float(sum(listt)/len(listt)))


# test_datagen = ImageDataGenerator(
#    rescale=1. / 255)
# #
# #
# test_generator = test_datagen.flow_from_directory(
#      TEST_DIR,
#      target_size=(HEIGHT, WIDTH),
#      batch_size=BATCH_SIZE,
#      class_mode='categorical')
# #
# fnames = test_generator.filenames
# # # # Get the ground truth from generator
# ground_truth = test_generator.classes
# # #
# # # # Get the label to class mapping from the generator
# label2index = test_generator.class_indices
# # #
# # #
# # # # Getting the mapping from class index to class label
# idx2label = dict((v, k) for k, v in label2index.items())
# # #
# # #
# predictions = loaded_model.predict_generator(test_generator,
#                                        steps=test_generator.samples / test_generator.batch_size, verbose=1)
# predicted_classes = np.argmax(predictions, axis=1)
# # # # print(predicted_classes)
# #
# errors = np.where(predicted_classes != ground_truth)[0]
# # print(errors)
# print("No of errors = {}/{}".format(len(errors),test_generator.samples))


# for j in range(len(errors)):
#     pred_class = np.argmax(predictions[errors[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}, filename: {}'.format(
#         fnames[errors[i]].split('/')[0],
#         pred_label,
#         predictions[errors[i]][pred_class], fnames[i])
#     img = image.load_img('/Users/Christos/PycharmProjects/kerasnew/Christos/EXPERIMENT/' + '',
#                          target_size=(224, 224))
#     preds = predict(loaded_model, img)
#     # #




    # original = load_img('{}/{}'.format(TEST_DIR, fnames[errors[i]]))
    # plt.figure(figsize=[9, 9])
    # plt.axis('off')
    # plt.title(title)
    # plt.imshow(original)
    # plt.show()
#
# for i in range(len(errors)):
#     pred_class = np.argmax(predictions[errors[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}, filename: {}'.format(
#         fnames[errors[i]].split('/')[0],
#         pred_label,
#         predictions[errors[i]][pred_class], fnames[i])
#     print(fnames[errors[i]],predictions[errors[i]])