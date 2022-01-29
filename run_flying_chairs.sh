N_SELECTED=1000
python flying_chairs.py --create_plot=False --n_selected=$N_SELECTED --model_name="flownet" --patch_mode="random"
python flying_chairs.py --create_plot=False --n_selected=$N_SELECTED --model_name="flownet" --patch_mode="adversarial"
python flying_chairs.py --create_plot=False --n_selected=$N_SELECTED --model_name="pwc" --patch_mode="random"
python flying_chairs.py --create_plot=False --n_selected=$N_SELECTED --model_name="pwc" --patch_mode="adversarial"
python flying_chairs.py --create_plot=False --n_selected=$N_SELECTED --model_name="raft" --patch_mode="random"
python flying_chairs.py --create_plot=False --n_selected=$N_SELECTED --model_name="raft" --patch_mode="adversarial"
