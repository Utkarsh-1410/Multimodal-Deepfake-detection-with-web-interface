from collections import OrderedDict
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from utils import *
from data_utils.utils import *
import json
import multiprocessing
from tqdm import tqdm
from data_utils.datasets import *
from torchvision.transforms import transforms


def get_face_detector_model(name='default'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # default is mtcnn model from facenet_pytorch
    if name == 'default':
        name = 'mtcnn'

    if name == 'mtcnn':
        detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)
    else:
        raise Exception("Unknown face detector model.")

    return detector


def locate_face_in_videofile(input_filepath=None, outfile_filepath=None):
    capture = cv2.VideoCapture(input_filepath)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))

    detector = get_face_detector_model()
    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        face_box = list(detector.detect(frame, landmarks=False))[0]
        if face_box is not None:
            for f in range(len(face_box)):
                fc = list(face_box[f])
                cv2.rectangle(frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
        frames.append(frame)
    create_video_from_images(frames, outfile_filepath, fps=org_fps, res=org_res)


def extract_landmarks_from_video(input_videofile, out_dir, batch_size=32, detector=None, overwrite=False, max_frames=100, sampling='uniform'):
    """
    Extract landmarks from video with intelligent frame sampling.
    
    Args:
        input_videofile: Path to input video
        out_dir: Output directory for landmark JSON
        batch_size: Batch size for face detection
        detector: Face detector model (MTCNN)
        overwrite: Whether to overwrite existing files
        max_frames: Maximum number of frames to process (default: 100)
        sampling: 'uniform' (evenly distributed) or 'sequential' (every Nth from start)
    """
    id = os.path.splitext(os.path.basename(input_videofile))[0]
    out_file = os.path.join(out_dir, "{}.json".format(id))

    if not overwrite and os.path.isfile(out_file):
        return

    capture = cv2.VideoCapture(input_videofile)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS not available
    video_duration = frames_num / fps if fps > 0 else 0
    
    if frames_num == 0:
        return
    
    if detector is None:
        detector = get_face_detector_model()

    # Adjust max_frames for short videos (< 10 seconds)
    # For short videos, process more frames proportionally or all frames
    if video_duration < 10 and frames_num < max_frames:
        # For videos < 10s, process all frames (they're short anyway)
        effective_max_frames = frames_num
    elif video_duration < 10:
        # For videos < 10s but with many frames, use proportional sampling
        # Aim for ~10 frames per second
        effective_max_frames = min(int(video_duration * 10), max_frames, frames_num)
    else:
        # For longer videos, use the standard max_frames
        effective_max_frames = max_frames

    # Determine which frames to process
    if sampling == 'uniform':
        # Evenly distribute frames across the entire video (better coverage)
        if frames_num <= effective_max_frames:
            frame_indices = list(range(frames_num))
        else:
            # Sample evenly spaced frames
            step = frames_num / effective_max_frames
            frame_indices = [int(i * step) for i in range(effective_max_frames)]
            # Ensure we get the last frame
            if frame_indices[-1] != frames_num - 1:
                frame_indices[-1] = frames_num - 1
    else:  # sequential
        # Process every Nth frame from start (faster but only covers beginning)
        frame_skip = max(1, frames_num // effective_max_frames)
        frame_indices = list(range(0, frames_num, frame_skip))[:effective_max_frames]

    frames_dict = OrderedDict()
    
    # Process selected frames
    for frame_idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.retrieve()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames_dict[frame_idx] = frame

    result = OrderedDict()
    batches = list()
    frames = list(frames_dict.values())
    num_frames_detected = len(frames)
    for i in range(0, num_frames_detected, batch_size):
        end = i + batch_size
        if end > num_frames_detected:
            end = num_frames_detected
        batches.append((list(range(i, end)), frames[i:end]))

    for j, frames_list in enumerate(batches):
        frame_indices, frame_items = frames_list
        batch_boxes, prob, keypoints = detector.detect(frame_items, landmarks=True)
        batch_boxes = [b.tolist() if b is not None else None for b in batch_boxes]
        keypoints = [k.tolist() if k is not None else None for k in keypoints]

        result.update({i: b for i, b in zip(frame_indices, zip(batch_boxes, keypoints))})

    with open(out_file, "w") as f:
        json.dump(result, f)


def my_collate(batch):
    batch = zip(*batch)
    return batch


def extract_landmarks_from_images_batch(input_images_dir, landmarks_file, batch_size=1024, detector=None,
                                        overwrite=True):
    if not overwrite and os.path.isfile(landmarks_file):
        return

    if detector is None:
        detector = get_face_detector_model()

    imgdata = SimpleImageFolder(input_images_dir, transforms_=None)
    data_loader = DataLoader(imgdata,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=multiprocessing.cpu_count(), collate_fn=my_collate)
    result = dict()
    for img_list in tqdm(data_loader):
        imgs, img_names = img_list
        # imgs = imgs.cuda()
        batch_boxes, prob, keypoints = detector.detect(imgs, landmarks=True)

        b = len(img_names)

        for j in range(b):
            boxes = None if batch_boxes[j] is None else batch_boxes[j].tolist()
            lm = None if keypoints[j] is None else keypoints[j].tolist()
            data = [boxes, lm]
            result.update({os.path.basename(img_names[j]): data})

    with open(landmarks_file, "w") as f:
        json.dump(result, f)


def extract_landmarks_from_video_batch(input_filepath_list, out_dir, batch_size=100, overwrite=False):
    """
    Extract landmarks from videos in batches.
    
    Args:
        input_filepath_list: List of video file paths
        out_dir: Output directory for landmark JSON files
        batch_size: Number of videos to process in each batch (default: 100)
        overwrite: Whether to overwrite existing landmark files
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Filter out already processed videos if not overwriting
    original_count = len(input_filepath_list)
    if not overwrite:
        remaining = []
        for video_path in input_filepath_list:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            out_file = os.path.join(out_dir, f"{video_id}.json")
            if not os.path.isfile(out_file):
                remaining.append(video_path)
        input_filepath_list = remaining
        skipped = original_count - len(remaining)
        if skipped > 0:
            print(f"Skipping {skipped} already processed videos")
    
    total_videos = len(input_filepath_list)
    if total_videos == 0:
        print("All videos already processed!")
        return
    
    print(f"Processing {total_videos} videos in batches of {batch_size}")
    
    # Process in batches
    num_batches = (total_videos + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_videos)
        batch_videos = input_filepath_list[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: Processing videos {start_idx + 1}-{end_idx} of {total_videos}")
        
    with multiprocessing.Pool(1) as pool:
        jobs = []
        results = []
        for input_filepath in batch_videos:
            jobs.append(pool.apply_async(extract_landmarks_from_video,
                                             (input_filepath, out_dir, 64, None, False, 100, 'uniform'),
                                         )
                        )

        for job in tqdm(jobs, desc=f"Batch {batch_idx + 1}/{num_batches}"):
            results.append(job.get())
        
        print(f"Completed batch {batch_idx + 1}/{num_batches}")
    
    print(f"\n✓ Finished processing all {total_videos} videos!")


def draw_landmarks_on_video(in_videofile, out_videofile, landmarks_file):
    capture = cv2.VideoCapture(in_videofile)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv2.CAP_PROP_FPS))
    with open(landmarks_file, 'r') as jf:
        face_box_dict = json.load(jf)
    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        if str(i) not in face_box_dict.keys():
            continue
        face_box = face_box_dict[str(i)]
        if face_box is not None:
            faces = face_box[0]
            lm = face_box[1]
            for f in range(len(faces)):
                fc = list(map(int, faces[f]))
                cv2.rectangle(frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
                keypoints = lm[f]
                cv2.circle(frame, tuple(map(int, keypoints[0])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[1])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[2])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[3])), 2, (0, 155, 255), 2)
                cv2.circle(frame, tuple(map(int, keypoints[4])), 2, (0, 155, 255), 2)

        frames.append(frame)
    create_video_from_images(frames, out_videofile, fps=org_fps, res=org_res)


def crop_faces_from_video(in_videofile, landmarks_path, crop_faces_out_dir, overwrite=False, frame_hops=10,
                          buf=0.10, clean_up=True):
    id = os.path.splitext(os.path.basename(in_videofile))[0]
    json_file = os.path.join(landmarks_path, id + '.json')
    out_dir = os.path.join(crop_faces_out_dir, id)
    if not os.path.isfile(json_file):
        return
    if not overwrite and os.path.isdir(out_dir):
        return

    try:
        with open(json_file, 'r') as jf:
            face_box_dict = json.load(jf)
    except Exception as e:
        print(f'failed to parse {json_file}')
        print(e)
        raise e

    os.makedirs(out_dir, exist_ok=True)
    capture = cv2.VideoCapture(in_videofile)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        if i % frame_hops != 0:
            continue
        success, frame = capture.retrieve()
        if not success or str(i) not in face_box_dict:
            continue

        crops = []
        bboxes = face_box_dict[str(i)][0]
        if bboxes is None:
            continue
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = int(h * buf)
            p_w = int(w * buf)
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            crops.append(crop)

        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(i, j)), crop)

    if clean_up and os.path.isfile(in_videofile):
        os.remove(in_videofile)


def crop_faces_from_image(in_image_path, landmarks_dict, crop_faces_out_dir, overwrite=True, buf=0.10):
    in_image_name = os.path.basename(in_image_path)
    in_image_id = os.path.splitext(in_image_name)[0]
    os.makedirs(crop_faces_out_dir, exist_ok=True)
    out_dir = crop_faces_out_dir
    os.makedirs(out_dir, exist_ok=True)

    if not overwrite and os.path.isdir(out_dir):
        return

    try:
        lm = landmarks_dict
        image = cv2.imread(in_image_path, cv2.IMREAD_COLOR)
        crops = list()
        bboxes = lm[0]
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = int(h * buf)
            p_w = int(w * buf)
            crop = image[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            crops.append(crop)

        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, "{}_{}.png".format(in_image_id, j)), crop)

    except Exception as e:
        pass


def crop_faces_from_image_batch(input_images_dir, landmarks_file, crop_faces_out_dir):
    os.makedirs(crop_faces_out_dir, exist_ok=True)

    with open(landmarks_file, 'r') as jf:
        landmarks_dict = json.load(jf)

    input_filepath_list = glob(input_images_dir + '/*')
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for input_filepath in tqdm(input_filepath_list, desc='Scheduling jobs'):
            in_image_name = os.path.basename(input_filepath)
            lm = landmarks_dict[in_image_name]
            jobs.append(pool.apply_async(crop_faces_from_image,
                                         (input_filepath, lm, crop_faces_out_dir,),
                                         )
                        )

        for job in tqdm(jobs, desc="Cropping faces from images"):
            results.append(job.get())


def crop_faces_from_video_batch(input_filepath_list, landmarks_path, crop_faces_out_dir, batch_size=100, overwrite=False):
    """
    Crop faces from videos in batches.
    
    Args:
        input_filepath_list: List of video file paths
        landmarks_path: Directory containing landmark JSON files
        crop_faces_out_dir: Output directory for cropped face images
        batch_size: Number of videos to process in each batch (default: 100)
        overwrite: Whether to overwrite existing cropped faces
    """
    os.makedirs(crop_faces_out_dir, exist_ok=True)
    
    total_videos = len(input_filepath_list)
    if total_videos == 0:
        print("No videos to process!")
        return
    
    print(f"Processing {total_videos} videos in batches of {batch_size}")
    
    # Process in batches
    num_batches = (total_videos + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_videos)
        batch_videos = input_filepath_list[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: Processing videos {start_idx + 1}-{end_idx} of {total_videos}")
        
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for input_filepath in batch_videos:
            jobs.append(pool.apply_async(crop_faces_from_video,
                                         (input_filepath, landmarks_path, crop_faces_out_dir,),
                                         )
                        )

        for job in tqdm(jobs, desc=f"Batch {batch_idx + 1}/{num_batches}"):
            results.append(job.get())
        
        print(f"Completed batch {batch_idx + 1}/{num_batches}")
    
    print(f"\n✓ Finished processing all {total_videos} videos!")


def extract_landmarks_for_datasets():
    #
    # Celeb-V2 dataset
    #

    landmarks_path = ConfigParser.getInstance().get_celeb_df_v2_landmarks_path()

    # Celeb-DF-v2 dataset (skip if paths don't exist)
    try:
        celeb_real_path = ConfigParser.getInstance().get_celeb_df_v2_real_path()
        if os.path.exists(celeb_real_path):
            print(f'Extracting landmarks from Celeb-df-v2 real data')
            input_filepath_list = glob(celeb_real_path + '/*')
            if input_filepath_list:
                extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)

        print(f'Extracting landmarks from Celeb-df-v2 fake data')
        data_path = ConfigParser.getInstance().get_celeb_df_v2_fake_path()
        if os.path.exists(data_path):
            input_filepath_list = glob(data_path + '/*')
            if input_filepath_list:
                extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)
    except Exception as e:
        print(f'Skipping Celeb-DF-v2 data: {e}')

    # YouTube-real data (skip if path doesn't exist)
    try:
        youtube_path = ConfigParser.getInstance().get_youtube_real_path()
        if os.path.exists(youtube_path):
            print(f'Extracting landmarks from YouTube-real data')
            youtube_landmarks_path = ConfigParser.getInstance().get_youtube_real_landmarks_path()
            os.makedirs(youtube_landmarks_path, exist_ok=True)
            input_filepath_list = glob(youtube_path + '/*')
            if input_filepath_list:
                extract_landmarks_from_video_batch(input_filepath_list, youtube_landmarks_path)
        else:
            print(f'Skipping YouTube-real data (path not found: {youtube_path})')
    except Exception as e:
        print(f'Skipping YouTube-real data: {e}')

    #
    # DFDC dataset (skip if paths don't exist)
    #
    try:
        dfdc_train_path = ConfigParser.getInstance().get_dfdc_train_data_path()
        if os.path.exists(dfdc_train_path):
            print(f'Extracting landmarks from DFDC train data')
            input_filepath_list = get_dfdc_training_video_filepaths(dfdc_train_path)
            landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
            os.makedirs(landmarks_path, exist_ok=True)
            extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)

            print(f'Extracting landmarks from DFDC valid data')
            data_path = ConfigParser.getInstance().get_dfdc_valid_data_path()
            if os.path.exists(data_path):
                input_filepath_list = glob(data_path + '/*')
                landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_valid_path()
                os.makedirs(landmarks_path, exist_ok=True)
                extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)

            print(f'Extracting landmarks from DFDC test data')
            data_path = ConfigParser.getInstance().get_dfdc_test_data_path()
            if os.path.exists(data_path):
                input_filepath_list = glob(data_path + '/*')
                landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_test_path()
                os.makedirs(landmarks_path, exist_ok=True)
                extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)
        else:
            print(f'Skipping DFDC data (path not found: {dfdc_train_path})')
    except Exception as e:
        print(f'Skipping DFDC data: {e}')

    # FDF data (skip if path doesn't exist)
    try:
        fdf_path = ConfigParser.getInstance().get_fdf_data_path()
        if os.path.exists(fdf_path):
            print(f'Extracting landmarks from FDF data')
            input_images_dir = fdf_path
            landmarks_file = ConfigParser.getInstance().get_fdf_json_path()
            os.makedirs(os.path.dirname(landmarks_file), exist_ok=True)
            extract_landmarks_from_images_batch(input_images_dir, landmarks_file, batch_size=32)
        else:
            print(f'Skipping FDF data (path not found: {fdf_path})')
    except Exception as e:
        print(f'Skipping FDF data: {e}')

    # FFHQ data (skip if path doesn't exist)
    try:
        ffhq_path = ConfigParser.getInstance().get_ffhq_data_path()
        if os.path.exists(ffhq_path):
            print(f'Extracting landmarks from FFHQ data')
            input_images_dir = ffhq_path
            landmarks_file = ConfigParser.getInstance().get_ffhq_json_path()
            os.makedirs(os.path.dirname(landmarks_file), exist_ok=True)
            extract_landmarks_from_images_batch(input_images_dir, landmarks_file, batch_size=128)
        else:
            print(f'Skipping FFHQ data (path not found: {ffhq_path})')
    except Exception as e:
        print(f'Skipping FFHQ data: {e}')

    # Custom dataset (process in batches)
    print(f'\n{"="*60}')
    print(f'Processing Custom Dataset (in batches)')
    print(f'{"="*60}')
    custom_landmarks_path = ConfigParser.getInstance().get_custom_dataset_landmarks_path()
    os.makedirs(custom_landmarks_path, exist_ok=True)
    
    print(f'\nExtracting landmarks from custom dataset REAL data...')
    data_path = ConfigParser.getInstance().get_custom_dataset_real_path()
    input_filepath_list = glob(data_path + '/*')
    if input_filepath_list:
        print(f'Found {len(input_filepath_list)} real videos')
        extract_landmarks_from_video_batch(input_filepath_list, custom_landmarks_path, batch_size=100, overwrite=False)
    else:
        print(f'No real videos found in {data_path}')

    print(f'\nExtracting landmarks from custom dataset FAKE data...')
    data_path = ConfigParser.getInstance().get_custom_dataset_fake_path()
    input_filepath_list = glob(data_path + '/*')
    if input_filepath_list:
        print(f'Found {len(input_filepath_list)} fake videos')
        extract_landmarks_from_video_batch(input_filepath_list, custom_landmarks_path, batch_size=100, overwrite=False)
    else:
        print(f'No fake videos found in {data_path}')


def crop_faces_for_datasets():
    #
    # Celeb-df-v2
    #
    landmarks_path = ConfigParser.getInstance().get_celeb_df_v2_landmarks_path()
    crops_path = ConfigParser.getInstance().get_celeb_df_v2_crops_train_path()

    print(f'Cropping faces from Celeb-df-v2 real data')
    data_path = ConfigParser.getInstance().get_celeb_df_v2_real_path()
    input_filepath_list = glob(data_path + '/*')
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    print(f'Cropping faces from Celeb-df-v2 fake data')
    data_path = ConfigParser.getInstance().get_celeb_df_v2_fake_path()
    input_filepath_list = glob(data_path + '/*')
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    # YouTube-real data
    print(f'Cropping faces from YouTube-real data')
    youtube_landmarks_path = ConfigParser.getInstance().get_youtube_real_landmarks_path()
    youtube_crops_path = ConfigParser.getInstance().get_youtube_real_crops_path()
    os.makedirs(youtube_crops_path, exist_ok=True)
    data_path = ConfigParser.getInstance().get_youtube_real_path()
    input_filepath_list = glob(data_path + '/*')
    crop_faces_from_video_batch(input_filepath_list, youtube_landmarks_path, youtube_crops_path)

    # Custom dataset (process in batches)
    print(f'\n{"="*60}')
    print(f'Cropping Faces from Custom Dataset (in batches)')
    print(f'{"="*60}')
    custom_landmarks_path = ConfigParser.getInstance().get_custom_dataset_landmarks_path()
    custom_crops_path = ConfigParser.getInstance().get_custom_dataset_crops_path()
    os.makedirs(custom_crops_path, exist_ok=True)
    
    print(f'\nCropping faces from custom dataset REAL data...')
    data_path = ConfigParser.getInstance().get_custom_dataset_real_path()
    input_filepath_list = glob(data_path + '/*')
    if input_filepath_list:
        print(f'Found {len(input_filepath_list)} real videos')
        crop_faces_from_video_batch(input_filepath_list, custom_landmarks_path, custom_crops_path, batch_size=100, overwrite=False)
    else:
        print(f'No real videos found in {data_path}')

    print(f'\nCropping faces from custom dataset FAKE data...')
    data_path = ConfigParser.getInstance().get_custom_dataset_fake_path()
    input_filepath_list = glob(data_path + '/*')
    if input_filepath_list:
        print(f'Found {len(input_filepath_list)} fake videos')
        crop_faces_from_video_batch(input_filepath_list, custom_landmarks_path, custom_crops_path, batch_size=100, overwrite=False)
    else:
        print(f'No fake videos found in {data_path}')

    #
    # DFDC dataset
    #
    print(f'Cropping faces from DFDC train data')
    data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
    input_filepath_list = get_dfdc_training_video_filepaths(data_path_root)
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_train_path()
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    print(f'Cropping faces from DFDC valid data')
    data_path = ConfigParser.getInstance().get_dfdc_valid_data_path()
    input_filepath_list = glob(data_path + '/*')
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_valid_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_valid_path()
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    print(f'Cropping faces from DFDC test data')
    data_path = ConfigParser.getInstance().get_dfdc_test_data_path()
    input_filepath_list = glob(data_path + '/*')
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_test_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_test_path()
    crop_faces_from_video_batch(input_filepath_list, landmarks_path, crops_path)

    #
    # FDF dataset
    #
    print(f'Extracting landmarks from FDF data')
    input_images_dir = ConfigParser.getInstance().get_fdf_data_path()
    landmarks_file = ConfigParser.getInstance().get_fdf_json_path()
    crops_path = ConfigParser.getInstance().get_fdf_crops_path()
    crop_faces_from_image_batch(input_images_dir, landmarks_file, crops_path)

    #
    # FFHQ dataset
    #
    print(f'Extracting landmarks from FFHQ data')
    input_images_dir = ConfigParser.getInstance().get_ffhq_data_path()
    landmarks_file = ConfigParser.getInstance().get_ffhq_json_path()
    crops_path = ConfigParser.getInstance().get_ffhq_crops_path()
    crop_faces_from_image_batch(input_images_dir, landmarks_file, crops_path)
