#!/bin/bash

#3 10 12 23 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45

v="v100"
OUTPUT_DIR="output_history/${v}"
mkdir -p ${OUTPUT_DIR}
for i in "../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img3.png" \
"../../SKING_DDJ_Dataset/entropydrop_website_generations/SKING_DDJ_v66/2RSZETA9PCRZHVEU_edited.png" \
"../../SKING_DDJ_Dataset/entropydrop_website_generations/SKING_DDJ_v66/33TVPML5UL2TKASR_edited.png" \
"../../SkingDataset/DDJ_real2render/test_output/unofficial_prompt1_1K_t41_51_66_71/img46.png" \
"../../SKING_DDJ_Dataset/entropydrop_website_generations/SKING_DDJ_v66/TWRLRRTHQP2UV368_edited.png" \
"../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img10.png" \
"../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img12.png" \
"../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img23.png" \
"../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img25.png" \
"../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img26.png" \
"../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img27.png" \
"../../SkingDataset/DDJ_real2render/test_output/official_prompt1_1K_t41_50_51_52/img28.png" \
"../../SKING_DDJ_Dataset/entropydrop_website_generations/SKING_DDJ_v54/1H6GAQA9BENWNB89_edited.png" \
"../../SKING_DDJ_Dataset/entropydrop_website_generations/SKING_DDJ_v54/PU418PLGERMFC487_edited.png"
do
    echo "Running ${i}"

    BASENAME=$(basename "$i" .png)
    OUTPUT_DIR2="${OUTPUT_DIR}/${BASENAME}"

    
    if [ -d "${OUTPUT_DIR2}" ]; then
        echo "Output directory ${OUTPUT_DIR2} already exists. Skipping..."
        continue
    fi

    PARSER_ONLY=true \
    FOREGROUND_FLOOD_TOLERANCE=0.09 \
    OUTER_UV_MIN_SOURCE_PIXELS=30 \
    CHECKPOINT="runs/dense_uv_parser_${v}/best.pt" \
    COMBINED="${i}" \
    ./run_infer.sh

    mv outputs ${OUTPUT_DIR2}
done

echo "All done!"