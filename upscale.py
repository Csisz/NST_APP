import replicate
import os
from dotenv import load_dotenv

load_dotenv()

def upscale_image(input_path):
    try:
        client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

        version = "cjwbw/real-esrgan:d0ee3d708c9b911f122a4ad90046c5d26a0293b99476d697f6bb7f2e251ce2d4"

        with open(input_path, "rb") as image_file:
            output_url = client.run(
                version,
                input={"image": image_file}
            )

        # ✅ Ensure it’s a string URL
        return str(output_url[0]) if isinstance(output_url, list) else str(output_url)

    except Exception as e:
        raise RuntimeError(f"Upscaling failed: {e}")
