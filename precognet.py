import matplotlib.pyplot as plt
from source.model.model import model
from source.model.normalizer import normalize_frames_to_model
from source.video.frames import extract_frames
from source.video.preprocessing import resizer
from source.configs import MODEL_IMAGE_SIZE

VIDEO_TEST = "../datasets/RWF-2000/train/Fight/_6-B11R9FJM_0.avi"

def main():
    # precog_model = model()
    # precog_model.summary()
    frames = resizer(extract_frames(VIDEO_TEST), MODEL_IMAGE_SIZE)
    normalized_frames = normalize_frames_to_model(frames)
    
    visible_frame = (normalized_frames*255).astype('uint8')
    plt.imshow(visible_frame[120])
    plt.show(block=True)


if __name__ == '__main__':
    main()