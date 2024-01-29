from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/0013035.jpg")

print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_totensor = trans_totensor(img)
writer.add_image("ToTensor", img_totensor)

# normalize
print(img_totensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_totensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)


# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))

#
img_resize = trans_resize(img)
print(img_resize)
print(type(img_resize))

#
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
# print(img_resize)

# compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(500, 1000)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)


writer.close()
