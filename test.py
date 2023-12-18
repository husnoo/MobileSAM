from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import datetime

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cpu" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)


ifname = "/home/nawal/IMG_20231210_220649_640x480.jpg"


image_bgr = cv2.imread(ifname)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# prompt
# predictor.set_image(image_rgb)
# masks, _, _ = predictor.predict("")

# segment everything
mask_generator = SamAutomaticMaskGenerator(mobile_sam)

start = datetime.datetime.now()
masks = mask_generator.generate(image_rgb)
stop = datetime.datetime.now()
print(stop-start)
print(masks)
