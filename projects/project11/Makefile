data_local:
	@if [ ! -d "./data/" ]; then \
		echo "did not find data folder, creating one"; \
		mkdir data; \
	fi
	@./load_hpc_local.sh
	unzip data/hotdog_nothotdog.zip -d ./data/
	rm data/hotdog_nothotdog.zip
