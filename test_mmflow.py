from mmflow.apis import inference_model, init_model

config_file = '/mmflow/configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.pth
checkpoint_file = 'mmflow-models/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
out = inference_model(model, '/mmflow/demo/frame_0001.png', '/mmflow/demo/frame_0002.png')
print(out.shape)


