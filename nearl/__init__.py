import requests
import re
import imghdr
import os


def pars_url(url_img):

    img = requests.get(url_img)
    string = str(img.content)
    array_strings = string.split('//')
    reg = r'jpg'
    reg2 = r'static'
    list_url = []

    for s in array_strings:
        if re.findall(reg, s) and re.findall(reg2, s):
            index = s.index('\\')

            list_url.append('http://' + s[:index])
    return list_url


def pars_image(list_url, train, amount, nam):
    exist_amount = amount

    name = nam
    global path_all
    global file

    x = 0
    number = 0
    while True:
        if exist_amount == 0:
            break

        try:

            path_all = train
            x += 1
            number += 1
            path = path_all + str(number) + nam +'_img.jpg'

            img = requests.get(list_url[x])

            if img.ok:
                file = open(path, 'wb')
                file.write(img.content)
                if imghdr.what(path) == 'jpeg':
                    print(str(number) + 'выглядит как JPEG картинка ' + name)

                    exist_amount -= 1

                else:
                    number -= 1
                    os.remove(path)

        except:
            number -= 1
            error = Exception
            print(error)
            file.close()

    file.close()


url_dog = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02087122'
url_cat = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02121620'
url_cat_zp = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02123159'
url_panda = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02510455'

path_dog_DT = '/home/pavel/Документы/datasets/animals/dog/'

#list_url = pars_url(url_dog)
#pars_image(list_url, path_dog_DT, amount=1000, nam='dog')

path_cat_DT = '/home/pavel/Документы/datasets/animals/cat/'

list_url = pars_url(url_cat_zp)
pars_image(list_url, path_cat_DT, amount=1000, nam='cat')

path_panda_DT = '/home/pavel/Документы/datasets/animals/panda/'

#list_url = pars_url(url_panda)
#pars_image(list_url, train=path_panda_DT, amount=1000, nam='panda')