
# coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
import torch.nn.functional as F
 
# 创建文件夹
if not os.path.exists('./standardgan_cifaimg'):
    os.mkdir('./standardgan_cifaimg')
 
 
def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 3, 32, 32)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out
 
 
batch_size = 128
num_epoch = 1000
z_dimension = 4096
# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # (x-mean) / std
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # (x-mean) / std
])
 
# mnist dataset mnist数据集下载
mnist = datasets.CIFAR10(
    root='./data_cifa/', train=True, transform=img_transform, download=True
)
 
# data loader 数据载入
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True
)
class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride,is_bn=True):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        if is_bn is True:
            self.add_module('norm',nn.BatchNorm2d(out_channel))
        # self.add_module('norm',nn.BatchNorm2d(out_channel)),
        #self.add_module('LeakyRelu',nn.LeakyReLU(0.2))
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
class ConvReComposeBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvReComposeBlock,self).__init__()
        self.add_module('reconv',nn.ConvTranspose2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        # self.add_module('norm',nn.BatchNorm2d(out_channel)),
        #self.add_module('LeakyRelu',nn.LeakyReLU(0.2))
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
    
 
# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
# class discriminator(nn.Module):
#     def __init__(self):
#         super(discriminator, self).__init__()
#         self.dis = nn.Sequential(
#             nn.Linear(3072, 256),  # 输入特征数为784，输出为256
#             nn.LeakyReLU(0.2),  # 进行非线性映射
#             nn.Linear(256, 256),  # 进行一个线性映射
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
#             # sigmoid可以班实数映射到【0,1】，作为概率值，
#             # 多分类用softmax函数
#         )
 
#     def forward(self, x):
#         x = self.dis(x)
#         return x
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.head = ConvBlock(3,64,3,1,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(2):
            block = ConvBlock(64,64,3,1,2)
            self.body.add_module('block%d'%(i+1),block)
        self.body1 =nn.Sequential()
        for jj in range(1):
            block = ConvBlock(64,16,3,1,2)
            self.body1.add_module('block%d'%(jj+1),block)
        self.tail = nn.Sequential(
            nn.Linear(256, 128),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(128, 16),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
        )
    def forward(self,x):
        x = self.head(x)
        #print(x.shape)
        x = self.body(x)
        #print(x.shape)
        x = self.body1(x)
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.tail(x)
        return x 
 
# ###### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
# class generator(nn.Module):
#     def __init__(self):
#         super(generator, self).__init__()
#         self.gen = nn.Sequential(
#             nn.Linear(100, 256),  # 用线性变换将输入映射到256维
#             nn.ReLU(True),  # relu激活
#             nn.Linear(256, 1024),  # 线性变换
#             nn.ReLU(True),  # relu激活
#             nn.Linear(1024, 3072)
#             ,  # 线性变换
#             nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
#         )
 
#     def forward(self, x):
#         x = self.gen(x)
#         return x
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        # self.before=nn.Sequential(nn.Linear(100, 4096),nn.LeakyReLU(0.2))
        self.head = ConvReComposeBlock(256,256,4,1,2) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)       
        self.body = nn.Sequential(
            ConvBlock(256,512,3,1,1),
            ConvBlock(512,1024,3,1,1),
            ConvReComposeBlock(1024,1024,4,1,2)
        )
        # for i in range(2):
        #     block = ConvBlock(512,512,3,1,1)
        #     self.body.add_module('block%d'%(i+1),block)
        # self.body.add_module('block',ConvReComposeBlock(512,1024,4,1,2))     
        self.body1 = nn.Sequential(
            ConvBlock(1024,512,3,1,1),
            ConvBlock(512,256,3,1,1),
            ConvReComposeBlock(256,256,4,1,2)
        )
        # for jj in range(2):
        #     block = ConvBlock(1024,1024,3,1,1)
        #     # block=ConvReComposeBlock(1024,512,4,1,2)
        #     self.body1.add_module('block%d'%(jj+1),block)
        # self.body.add_module('block',ConvReComposeBlock(1024,512,4,1,2))  
        self.tail = nn.Sequential(
            nn.Conv2d(256,64,3,1,1),
            nn.Conv2d(64,3,3,1,1),
            nn.Tanh()
            # nn.Sigmoid()
        )
    def forward(self,x):
        # x1=x
        # x_upsampling=F.interpolate(x1, scale_factor=8, mode='bicubic')#上采样2
        # print(x.shape)
        # x=self.before(x)
        # print(x.shape)
        x=x.reshape(x.shape[0],256,4,4)
        # print(x.shape)
        x = self.head(x)
        # print(x.shape)
        #x = F.interpolate(x, scale_factor=2, mode='bicubic')#上采样2
        x = self.body(x)
        # print(x.shape)
        x = self.body1(x)
        # print(x.shape)
        #x = F.interpolate(x, scale_factor=2, mode='bicubic')#上采样2
        x = self.tail(x)
        #x = F.interpolate(x, scale_factor=2, mode='bicubic')#上采样2
        return x
# 创建对象
D = discriminator()
G = generator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
 
 
# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
#criterion = nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0005, betas=[0.5, 0.9])
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0005, betas=[0.5, 0.9])
 
# ##########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        #print('the num i is {}'.format(i))
        # view()函数作用是将一个多行的Tensor,拼接成一行
        # 第一个参数是要拼接的tensor,第二个参数是-1
        # =============================训练判别器==================
        img = img.view(num_img, -1)  # 将图片展开为28*28=784
        real_img = Variable(img).cuda()  # 将tensor变成Variable放入计算图中
        real_img_pic=real_img.reshape(-1,3,32,32)
        real_label = Variable(torch.ones(num_img)).cuda()  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的图片的label为0
 
        # ########判别器训练train#####################
        # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
        # 计算真实图片的损失
        #for j in range(5):
        real_out = D(real_img_pic)  # 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        # 计算假的图片的损失
        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成一些噪声
        z2=z.reshape(-1,256,4,4)
        fake_img = G(z2)  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
        # print(fake_img.shape)
        fake_out = D(fake_img.detach())  # 判别器判断假的图片，
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数
 
        # ==================训练生成器============================
        # ###############################生成网络的训练###############################
        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        # 这样就达到了对抗的目的
        # 计算假的图片的损失
        # z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 得到随机噪声
        # z3=z.reshape(-1,3,4,4)
        # fake_img1 = G(z3)  # 随机噪声输入到生成器中，得到一副假的图片
        output = D(fake_img)  # 经过判别器得到的结果
        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
 
        # 打印中间的损失
        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
            ))
        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './standardgan_cifaimg/real_images.png')
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './standardgan_cifaimg/fake_images-{}.png'.format(epoch + 1))
 
# 保存模型
torch.save(G.state_dict(), './standardgan_cifaimg/generator.pth')
torch.save(D.state_dict(), './standardgan_cifaimg/discriminator.pth')
