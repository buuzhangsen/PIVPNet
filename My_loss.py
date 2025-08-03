import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# __all__ = ['PSNRLoss', 'CharbonnierLoss', 'PerceptualLoss', 'PatchDiscriminator', 'GANLoss']


# class PSNRLoss(nn.Module):
# 	def __init__(self, eps=1e-8):
# 		super(PSNRLoss, self).__init__()
# 		self.eps = eps

# 	def forward(self, output, target):
# 		diff = output - target
# 		mse = (diff * diff).mean(dim=(1, 2, 3))
# 		loss = torch.log(mse + self.eps).mean()
# 		return loss

class ColorGradientLoss(nn.Module):
    def __init__(self, loss_type='l1', epsilon=1e-3):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.epsilon = epsilon

    def compute_gradient(self, x):
        # 计算梯度（确保与输入x的尺寸对齐）
        h_grad = x[:, :, :, :-1] - x[:, :, :, 1:]  # 输出尺寸: [B, C, H, W-1]
        v_grad = x[:, :, :-1, :] - x[:, :, 1:, :]  # 输出尺寸: [B, C, H-1, W]
        
        if self.loss_type == 'l1':
            h_grad = torch.abs(h_grad)
            v_grad = torch.abs(v_grad)
        elif self.loss_type == 'l2':
            h_grad = torch.pow(h_grad, 2)
            v_grad = torch.pow(v_grad, 2)
        return h_grad, v_grad

    def forward(self, gen_imgs, real_imgs):
        # 生成图像梯度
        gen_h_grad, gen_v_grad = self.compute_gradient(gen_imgs)  # [B,C,H,W-1] 和 [B,C,H-1,W]
        
        # 真实图像梯度
        real_h_grad, real_v_grad = self.compute_gradient(real_imgs)  # [B,C,H,W-1] 和 [B,C,H-1,W]
        
        # 确保维度对齐（关键修复步骤）
        # 水平梯度损失：裁剪生成和真实图像的高度/宽度至最小公共尺寸
        min_h = min(gen_h_grad.size(2), real_h_grad.size(2))
        min_w_hgrad = min(gen_h_grad.size(3), real_h_grad.size(3))
        gen_h_grad = gen_h_grad[:, :, :min_h, :min_w_hgrad]
        real_h_grad = real_h_grad[:, :, :min_h, :min_w_hgrad]
        h_loss = torch.mean(torch.abs(gen_h_grad - real_h_grad))
        
        # 垂直梯度损失：同理
        min_v = min(gen_v_grad.size(2), real_v_grad.size(2))
        min_w_vgrad = min(gen_v_grad.size(3), real_v_grad.size(3))
        gen_v_grad = gen_v_grad[:, :, :min_v, :min_w_vgrad]
        real_v_grad = real_v_grad[:, :, :min_v, :min_w_vgrad]
        v_loss = torch.mean(torch.abs(gen_v_grad - real_v_grad))
        
        total_loss = (h_loss + v_loss) / 2
        return total_loss
    
class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0, eps=1e-8):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val  # 图像像素的最大值（例如 [0, 1] 对应 1.0，[0, 255] 对应 255）
        self.eps = eps

    def forward(self, output, target):
        # 计算 MSE
        mse = torch.mean((output - target) ** 2, dim=(1, 2, 3))
        # 计算 PSNR 的对数形式（避免数值溢出）
        psnr = 10 * torch.log10((self.max_val ** 2) / (mse + self.eps))
        # 将 PSNR 转换为损失函数（越小越好）
        loss = -psnr.mean()
        return loss

class CharbonnierLoss(nn.Module):
	def __init__(self, eps=1e-3):
		super(CharbonnierLoss, self).__init__()
		self.eps2 = eps ** 2

	def forward(self, output, target):
		diff = output - target
		loss = torch.mean(torch.sqrt((diff * diff) + self.eps2))
		return loss


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super().__init__()
		vgg_pretrained_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h_relu1 = self.slice1(X)
		h_relu2 = self.slice2(h_relu1)
		h_relu3 = self.slice3(h_relu2)
		h_relu4 = self.slice4(h_relu3)
		h_relu5 = self.slice5(h_relu4)
		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
		return out


# Perceptual loss that uses a pretrained VGG network
class PerceptualLoss(nn.Module):
	def __init__(self):
		super(PerceptualLoss, self).__init__()
		self.vgg = VGG19().cuda()
		self.criterion = nn.L1Loss()
		self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

	def forward(self, x, y):
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)
		loss = 0
		for i in range(len(x_vgg)):
			loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
		return loss

# PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
	def __init__(self):
		super(PatchDiscriminator, self).__init__()
		
		def discriminator_block(in_filters, out_filters, normalize=True):
			"""Returns downsampling layers of each discriminator block"""
			layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
			if normalize:
				layers.append(nn.InstanceNorm2d(out_filters))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*discriminator_block(3, 64, normalize=False),
			*discriminator_block(64, 128),
			*discriminator_block(128, 256),
			*discriminator_block(256, 512),
			nn.ZeroPad2d((1, 0, 1, 0)),
			nn.Conv2d(512, 1, 4, padding=1)
		)

	def forward(self, input):
		return self.model(input)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
	def __init__(self, gan_mode='ls', target_real_label=1.0, target_fake_label=0.0,
				 tensor=torch.FloatTensor, opt=None):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_tensor = None
		self.fake_label_tensor = None
		self.zero_tensor = None
		self.Tensor = tensor
		self.gan_mode = gan_mode
		self.opt = opt
		if gan_mode == 'ls':
			pass
		elif gan_mode == 'original':
			pass
		elif gan_mode == 'w':
			pass
		elif gan_mode == 'hinge':
			pass
		else:
			raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			if self.real_label_tensor is None:
				self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
				self.real_label_tensor.requires_grad_(False)
			return self.real_label_tensor.expand_as(input)
		else:
			if self.fake_label_tensor is None:
				self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
				self.fake_label_tensor.requires_grad_(False)
			return self.fake_label_tensor.expand_as(input)

	def get_zero_tensor(self, input):
		if self.zero_tensor is None:
			self.zero_tensor = self.Tensor(1).fill_(0)
			self.zero_tensor.requires_grad_(False)
		return self.zero_tensor.expand_as(input)

	def loss(self, input, target_is_real, for_discriminator=True):
		if self.gan_mode == 'original':  # cross entropy loss
			target_tensor = self.get_target_tensor(input, target_is_real)
			loss = F.binary_cross_entropy_with_logits(input, target_tensor)
			return loss
		elif self.gan_mode == 'ls':
			target_tensor = self.get_target_tensor(input, target_is_real)
			return F.mse_loss(input, target_tensor)
		elif self.gan_mode == 'hinge':
			if for_discriminator:
				if target_is_real:
					minval = torch.min(input - 1, self.get_zero_tensor(input))
					loss = -torch.mean(minval)
				else:
					minval = torch.min(-input - 1, self.get_zero_tensor(input))
					loss = -torch.mean(minval)
			else:
				assert target_is_real, "The generator's hinge loss must be aiming for real"
				loss = -torch.mean(input)
			return loss
		else:
			# wgan
			if target_is_real:
				return -input.mean()
			else:
				return input.mean()

	def __call__(self, input, target_is_real, for_discriminator=True):
		# computing loss is a bit complicated because |input| may not be
		# a tensor, but list of tensors in case of multiscale discriminator
		if isinstance(input, list):
			loss = 0
			for pred_i in input:
				if isinstance(pred_i, list):
					pred_i = pred_i[-1]
				loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
				bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
				new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
				loss += new_loss
			return loss / len(input)
		else:
			return self.loss(input, target_is_real, for_discriminator)