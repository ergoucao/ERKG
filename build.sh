mkdir knowledge4tsf
conda create -n knowledge4tsf
conda activate knowledge4tsf
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pandas
conda install scikit-learn
conda install -c conda-forge einops
conda install transformers
conda install -c conda-forge  wandb


/home/cx/miniconda3/envs/knowledge4tsf/bin/python3 -u /home/cx/projects/knowledge4tsf/main.py --model 3 --device cuda:0 --tsf_model patchTsMixer --pred_len 1 --status_file status_model1_predL_1_horizon_1_topk_22_umass3.pth --data umass3 --data_dim 22 --topk 22 >/tmp/umass3_1_patchTsMixer.log