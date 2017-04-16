import os
data = open("data.txt","w+")
train = open("train.txt","w+")
test = open("test.txt","w+")

im_dir = "CroppedYale/"
f = os.listdir(im_dir)
folders = [None] * 13
for i in range(len(folders)):
    folders[i] = int(f[i].split("B")[1])
    t = os.listdir(im_dir + f[i])
    for j in range(len(t)):
        imname = im_dir + f[i] + "/" + t[j] + "\t" + str(i + 1) + "\n"
        data.write(imname)
        if j > 55:
            if("Ambient" not in imname):
                test.write(imname)
        else:
            if("Ambient" not in imname):
                train.write(imname)

#    for j in range()

data.close()
train.close()
test.close()
