# ============ #
# Build images #
# ============ #

ifndef USER_NAME
$(error USER_NAME not defined)
endif

dexnet:
	docker build --rm \
	    --build-arg from=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 \
	    --tag $(USER_NAME)/benchmark_dexnet:latest \
	    ../dexnet/

gpd:
	docker build --rm \
	    --build-arg from=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 \
	    --tag $(USER_NAME)/benchmark_gpd:latest \
	    ../gpd/

superquadric:
	docker build --rm \
	    --build-arg from=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 \
	    --tag $(USER_NAME)/benchmark_superquadrics:latest \
	    ../superquadrics_based/

6dgraspnet:
	docker build --rm \
		--build-arg from=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 \
		--tag $(USER_NAME)/benchmark_6dgraspnet:latest \
		../6dgraspnet/
