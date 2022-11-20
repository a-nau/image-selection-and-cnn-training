input_path="${PWD}/data/scraped/01_raw"
output_path="${PWD}/data/scraped/03_removed_bg"
mkdir -p "${output_path}"

# Execute default
docker run -it --rm --name remb --cpus 40 \
--mount type=bind,source="${output_path}",target=/rembg/data/output \
--mount type=bind,source="${input_path}",target=/rembg/data/input \
danielgatis/rembg p ./data/input ./data/output