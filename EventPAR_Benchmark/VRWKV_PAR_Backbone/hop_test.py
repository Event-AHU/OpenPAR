# from models.vit import *
# model_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/songhaoyu/PAR/VRWKV_PAR/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
# vit = vit_base()
# vit.load_param(model_path)
# img_temps = torch.randn(16, 5, 3, 256, 128)
# b_s = img_temps.size(0)
# stored_pattern=[vit(img_temps[i,:,:]) for i in range(b_s)]
# for i in range(b_s):
#     print(stored_pattern[i].shape)


from models.vrwkv import *
device = torch.device("cuda")   
vrwkv = VRWKV()
imgs = torch.randn(16, 3, 224, 224)
features = vrwkv.forward(imgs).to(device).float()
img_temps = torch.randn(16, 5, 3, 256, 128)
stored_pattern=[vrwkv.forward(img_temps[i,:,:]) for i in range(16)]
