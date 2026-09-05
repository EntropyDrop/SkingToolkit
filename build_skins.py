import json
import io
import zstandard as zstd
import base64
from mc_skin_utils.ensure_skin64x64 import convert_skin_64x32_to_64x64
from mc_skin_utils.alice_to_steve import alice_to_steve
from PIL import Image
from mc_skin_utils.clean_skins_extra_pixels import clean_skin
from mc_skin_utils.validator import validate_base_layer, is_alex

i = 0
def extract_from_zst(f):
    global i
    with open(f, "rb") as f:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text_stream:
                obj = json.loads(line)
                fname = f'skins/{i}.png'
                print(i)

                image_data = base64.b64decode(obj["image"])

                try:
                    with Image.open(io.BytesIO(image_data)) as img:
                        skin_img = img.copy()
                except Exception as e:
                    print(f"Error processing image {i}: {e}")
                    continue

                if skin_img.size == (64, 32):
                    #skin_img = convert_skin_64x32_to_64x64(skin_img)
                    continue
                if is_alex(skin_img):
                    skin_img = alice_to_steve(skin_img)
                if not validate_base_layer(skin_img, is_alex=False):
                    continue

                cleaned_skin_img = clean_skin(skin_img)
                cleaned_skin_img.save(fname)
                i+=1

extract_from_zst("../Minecraft-Skins-20M/dataset_0000.jsonl.zst")
extract_from_zst("../Minecraft-Skins-20M/dataset_0001.jsonl.zst")
extract_from_zst("../Minecraft-Skins-20M/dataset_0002.jsonl.zst")
extract_from_zst("../Minecraft-Skins-20M/dataset_0003.jsonl.zst")
extract_from_zst("../Minecraft-Skins-20M/dataset_0004.jsonl.zst")
extract_from_zst("../Minecraft-Skins-20M/dataset_0005.jsonl.zst")
extract_from_zst("../Minecraft-Skins-20M/dataset_0006.jsonl.zst")