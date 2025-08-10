import os
import sys
import importlib.util

def patch_basicsr():
    try:
        spec = importlib.util.find_spec("basicsr")
        if spec is None:
            print("basicsr not found. Make sure it's installed.")
            return

        basicsr_path = os.path.dirname(spec.origin)
        degradation_file = os.path.join(basicsr_path, "data", "degradations.py")

        if not os.path.isfile(degradation_file):
            print("Could not find degradations.py in basicsr.")
            return

        with open(degradation_file, "r") as f:
            content = f.read()

        if "functional_tensor" in content:
            patched = content.replace(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                "from torchvision.transforms.functional import rgb_to_grayscale"
            )

            with open(degradation_file, "w") as f:
                f.write(patched)

            print(f"Successfully patched: {degradation_file}")
        else:
            print("No patching needed. Already fixed.")

    except Exception as e:
        print(f"Error during patch: {e}")

if __name__ == "__main__":
    patch_basicsr()
