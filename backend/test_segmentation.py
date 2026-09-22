from app.segmentation import DeepLabSegmenter


segmenter = DeepLabSegmenter(device="cpu")

result = segmenter.predict(
    "outputs/1fecdf1a26034adeacbc46de9bde764a_sr.tif",
    "outputs/1fecdf1a26034adeacbc46de9bde764a_segmentation.tif",
)

print(result)