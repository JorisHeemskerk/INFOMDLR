import cv2
import numpy as np
import torch
import ffmpeg
from torchvision import transforms
from typing import Callable

from create_logger import create_logger
from decode import decode_predictions
from visualise import draw_boxes
from yolov1_resnet import YOLOv1ResNet


def process_batch(
    writer: cv2.VideoWriter,
    frames: list[np.ndarray],
    model: torch.nn.Module,
    transform: Callable,
    grid_size: int,
    conf_threshold: float,
    device: str,
) -> list[np.ndarray]:
    """
    Makes predictions on a batch of frames and writes the detections to
    a video.

    :param writer: Writer object the processed frames can be written to.
    :type writer: cv2.VideoWriter
    :param frames: Input frames.
    :type frames: list[np.ndarray]
    :param model: Trained model.
    :type model: torch.nn.Module
    :param img_size: Square size that the model accepts.
    :type img_size: int
    :param grid_size: YOLO grid size.
    :type grid_size: int
    :param conf_threshold: Objectness confidence threshold.
    :type conf_threshold: float
    :param device: Device to move data to.
    :type device: str
    """
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    batch_tensor = torch.stack([
        transform(rgb) for rgb in rgb_frames
    ]).to(device)
 
    with torch.no_grad():
        output = model(batch_tensor)
 
    decoded = decode_predictions(output.view(-1, grid_size, grid_size, 7))
 
    for i, rgb in enumerate(rgb_frames):
        annotated_rgb = draw_boxes(
            rgb.astype(np.float32) / 255.0, 
            tuple(t[i] for t in decoded),
            conf_threshold,
            ["cat", "dog"],
            5,
            1,
            2
        )
 
        writer.write(
            cv2.cvtColor(
                (annotated_rgb * 255).clip(0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
        )

def process_video(
    model_path: str,
    video_path: str,
    output_path: str,
    img_size: int,
    grid_size: int,
    conf_threshold: float,
    batch_size: int,
) -> None:
    """
    Load a trained model and produce an annotated copy of a video.

    :param model_path: Path to the `.pth` model file.
    :type model_path: str
    :param video_path: Path to the source video file.
    :type video_path: str
    :param output_path: Path to the annotated copy of the video.
    :type output_path: str
    :param img_size: Square size that the model accepts.
    :type img_size: int
    :param grid_size: YOLO grid size.
    :type grid_size: int
    :param conf_threshold: Objectness confidence threshold.
    :type conf_threshold: float
    :param batch_size: Maximum number of frames to process before
        reading new frames.
    :type batch_size: int
    """
    logger = create_logger("predict_video")

    # Initialise Device.
    device = torch.accelerator.current_accelerator().type if \
        torch.accelerator.is_available() else "cpu"
    logger.info(f"Using {device} device")

    # Load the model.
    model = YOLOv1ResNet.load(model_path, logger)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from: {model_path}")

    # Transform the video input into yolo input size.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Video reader.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Video: {orig_w}x{orig_h} @ {fps:.2f} fps | "
        f"{total_frames} frames total"
    )

    # Video writer.
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (orig_w, orig_h)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not write to: {output_path}")

    logger.info(f"Writing video to: {output_path}")

    # Loop over the video and write to output video.
    frame_buffer = []
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == batch_size:
            process_batch(
                writer,
                frame_buffer,
                model,
                transform,
                grid_size,
                conf_threshold,
                device
            )
            
            frame_buffer.clear()
            frames_processed += batch_size
            logger.debug(
                f"Processed {frames_processed}/{total_frames} frames "
                f"({100 * frames_processed / max(total_frames, 1):.1f}%)"
            )

    if frame_buffer:
        process_batch(
            writer,
            frame_buffer,
            model,
            transform,
            grid_size,
            conf_threshold,
            device
        )

    cap.release()
    writer.release()

    # Audio handling.
    ffmpeg.output(
        ffmpeg.input(output_path).video,
        ffmpeg.input(video_path).audio,
        output_path.replace(".mp4", "_audio.mp4")
    ).run()

    logger.info(f"Done making the video.")


if __name__ == "__main__":
    process_video(
        model_path="assignment_4/models/resnet_model.pth",
        video_path="assignment_4/data/cool_video_cropped.mp4",
        output_path="assignment_4/output/processed_video.mp4",
        img_size=224,
        grid_size=7,
        conf_threshold=.5,
        batch_size=195,
    )
