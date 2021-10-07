
import cv2

with open("./dataset/val.txt", 'r') as txt:
    lines = txt.readlines()

print(len(lines))
print("***")
print()
img384 = list()
imgnone = list()

for line in lines:
    img_path = line.strip('\n')
    # print(img_path)
    img = cv2.imread(img_path)
    # print(img.shape)
    # print(type(img))
    if img is None:
        imgnone.append(img_path)
    elif (img.shape[0] != 192):
        img384.append((img_path, img.shape))

print(len(imgnone))
print(imgnone)
print("***")
print()
print(len(img384))
print(len(lines) - len(img384) - len(imgnone))
# print(img384)
