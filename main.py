import ray
from model.gemma_model_vllm import LLMPredictor
from vllm import SamplingParams, sampling_params

## Data Paths
BASE_PATH = 's3://anyscale-customer-dataplane-data-production-us-east-2/artifact_storage/org_6687q89lgh27q3z41zesm2fsq6/cld_j25ipm5kli358v41pn9c96gjg3/BurberryData:john_:kpbdm'
IMG_PATH = BASE_PATH + "/images"
DATA_PATH = BASE_PATH + "/data"
CAPTION_PATH = BASE_PATH + "/captions/2"

# The number of LLM instances to use.
num_llm_instances = 4
# The number of GPUs to use per LLM instance.
num_gpus_per_instance = 1
# Number of images to process for testing
LIMIT = 3000
# Accelerator Type
accelerator_type = "A10G"
# Batch size
batch_size = 50
# Sampling params
sampling_params = SamplingParams(temperature=0.1)

# Ray data pipeline
img_data = ray.data.read_images(IMG_PATH, include_paths=True).limit(LIMIT)
ds = (
    img_data
    .map_batches(
        LLMPredictor,
        concurrency=num_llm_instances,
        num_gpus=num_gpus_per_instance,    
        batch_size=batch_size,
        accelerator_type=accelerator_type,
        fn_constructor_kwargs={"image_col": "image", "sampling_params": sampling_params}
    )
)

ds.write_parquet(
        path=CAPTION_PATH,
        try_create_dir=False
)