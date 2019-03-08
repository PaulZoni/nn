from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('/home/pavel/Документы/datasets/generator/i/IMG_20190308_171238.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

for batch in datagen.flow(x, batch_size=1, save_to_dir='/home/pavel/Документы/datasets/generator/much_i/', save_prefix='3cat_img', save_format='jpg'):
    i += 1
    if i > 11:
        break