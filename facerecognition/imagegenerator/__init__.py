import os

directory = '/home/pavel/Документы/datasets/person/'

contdir = []

for i in os.walk(directory):
    contdir.append(i)

counter = 0

path_to = '/home/pavel/Документы/datasets/face/persons/'


for i in contdir:

    for elem in i[2]:
        file_to = open(path_to + str(counter) +'img.jpg', 'wb')
        file_from = open(i[0] + '/' + elem, 'rb')
        image = file_from.read()
        file_to.write(image)
        print('img:' + str(counter))
        if counter == 5000:
            break

    counter += 1


