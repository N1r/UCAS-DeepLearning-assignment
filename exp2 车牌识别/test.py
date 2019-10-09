from PIL import Image
img = Image.open("./1.bmp")
print(img.getpixel((0,0)))
img = img.resize((20,20))
for i in range(20):
    for j in range(20):
        print(img.getpixel((i,j)))


