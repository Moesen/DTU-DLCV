ffhq_weights:
	wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl -P models/

styleclip_weights:
	wget https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt -P StyleCLIP/

align_images:
	python FFHQ-Alignment/FFHQ-Alignmnet/ffhq-align.py -s data/raw -d data/processed/ -o 1024 -t 1024

