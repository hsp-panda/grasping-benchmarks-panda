# ============ #
# Build images #
# ============ #

ifndef USER_NAME
$(error USER_NAME not defined)
endif

dexnet:
	DOCKER_BUILDKIT=1 docker build --ssh default --rm \
	    --build-arg from=ghcr.io/hsp-panda/nvidia/cudagl-10.0-cudnn7-devel-ubuntu18.04:latest \
	    --tag $(USER_NAME)/benchmark_dexnet:latest \
	    ../dexnet/

gpd:
	DOCKER_BUILDKIT=1 docker build --ssh default --rm \
		--build-arg from=hsp-panda/nvidia/cudagl-10.2-cudnn7-devel-ubuntu18.04:latest \
	    --tag $(USER_NAME)/benchmark_gpd:latest \
	    ../gpd/

superquadrics:
	DOCKER_BUILDKIT=1 docker build --ssh default --rm \
		--build-arg from=ghcr.io/hsp-panda/nvidia/cudagl-10.0-cudnn7-devel-ubuntu18.04:latest \
	    --tag $(USER_NAME)/benchmark_superquadrics:latest \
	    ../superquadrics_based/

6dgraspnet:
	docker build --rm \
		--build-arg from=ghcr.io/hsp-panda/nvidia/cudagl-10.0-cudnn7-devel-ubuntu18.04:latest \
		--tag $(USER_NAME)/benchmark_6dgraspnet:latest \
		../6dgraspnet/
