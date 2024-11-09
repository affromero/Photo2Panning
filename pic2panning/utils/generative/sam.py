"""contains utility functions for object detection and segmentation."""

import gc
from typing import Any, Literal

import numpy as np
import torch
from jaxtyping import Bool
from PIL import Image
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import pipeline
from transformers.pipelines.base import Pipeline

from pic2panning.utils.logger import get_logger

logger = get_logger()


@dataclass(config=ConfigDict(extra="forbid"))
class BoundingBox:
    """A class representing a bounding box in an image."""

    xmin: float
    """The minimum X-coordinate of the bounding box."""
    ymin: float
    """The minimum Y-coordinate of the bounding box."""
    xmax: float
    """The maximum X-coordinate of the bounding box."""
    ymax: float
    """The maximum Y-coordinate of the bounding box."""

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Calculate the minimum and maximum X and Y coordinates of the.

            bounding box.

        This method from the 'BoundingBox' class generates a tuple
            containing the X and Y coordinates
        of the minimum and maximum points of the bounding box.

        Arguments:
            None
        Returns:
            Tuple[float, float, float, float]: A tuple in the format (min_x,
                min_y, max_x, max_y)
            representing the minimum and maximum X and Y coordinates of the
                bounding box.

        Example:
            >>> bbox = BoundingBox(...)
            >>> bbox.xyxy()
            (min_x, min_y, max_x, max_y)

        Note:
            This method does not take any arguments. It calculates the
                coordinates based on the
            properties of the 'BoundingBox' instance.

        """
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Calculate and return the coordinates and dimensions of a bounding.

            box.

        This function does not take any arguments. It calculates and returns
            a tuple containing
        the x, y coordinates and width, height dimensions of a bounding box.

        Returns:
            Tuple[int, int, int, int]: A tuple containing the x, y
                coordinates and width, height
            dimensions of the bounding box. The values are ordered as (x, y,
                width, height).

        Example:
            >>> get_bounding_box()
            (10, 20, 30, 40)

        Note:
            The function assumes that the bounding box is calculated based
                on some predefined conditions.

        """
        return (
            self.xmin,
            self.ymin,
            self.xmax - self.xmin,
            self.ymax - self.ymin,
        )

    def __str__(self) -> str:
        """Return a string representation of the BoundingBox object.

        This method generates a string that represents the BoundingBox
            object by displaying its minimum and maximum x and y values. No
            arguments are required for this method.

        Returns:
            str: A string representation of the BoundingBox object. The
                string includes the minimum and maximum x and y values.

        Example:
            >>> bbox = BoundingBox(0, 0, 1, 1)
            >>> print(bbox)
            'BoundingBox: Min(x=0, y=0), Max(x=1, y=1)'
        Note:
            This method is typically used for debugging purposes.

        """
        return f"xmin: {self.xmin}, ymin: {self.ymin}, xmax: {self.xmax}, ymax: {self.ymax}"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class DetectionResult:
    """A class representing the result of an object detection operation."""

    score: float
    """The confidence score of the detected label."""
    label: str
    """The label detected in the image."""
    box: BoundingBox
    """The bounding box coordinates of the detected label."""

    @classmethod
    def from_dict(cls, detection_dict: dict[str, Any]) -> "DetectionResult":
        """Construct a DetectionResult object from a dictionary.

        This class method takes a dictionary containing detection
            information
        and returns a DetectionResult object with the attributes extracted
        from the input dictionary.

        Arguments:
            detection_dict (dict): A dictionary containing detection
                information
            such as score, label, and bounding box coordinates.

        Returns:
            DetectionResult: A DetectionResult object with the specified
            attributes extracted from the input dictionary.

        Example:
            >>> DetectionResult.from_dict(detection_dict)

        """
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


def detect(
    image: Image.Image,
    *,
    labels: list[str],
    object_detector: Pipeline,
    threshold: float = 0.3,
) -> list[DetectionResult]:
    """Use Grounding DINO to detect labels in an image in a zero-shot fashion.

    This function applies the Grounding DINO model to an input image to
        detect
    a set of labels without any prior training (zero-shot).

    Arguments:
        image (arg1_type): The input image where labels have to be detected.
            This should be a 2D or 3D array representing an image.
        labels (arg2_type): A list of labels that the function should try to
            detect in the image.
        object_detector (arg3_type): The object detector pipeline used for
            zero-shot object detection.
        threshold (float, optional): The confidence threshold for the
            detected labels. Defaults to 0.3.

    Returns:
        list: A list of labels detected in the image. Each element in the
            list is a tuple where the first element is the label and the
            second element is the confidence score.

    Example:
        >>> detect_labels(image, ["cat", "dog"], model=GroundingDINO())

    Note:
        The Grounding DINO model is a powerful tool for zero-shot image
            label detection. However, the accuracy of the detection can
            depend on the quality of the input image and the relevance of
            the labels.

    """
    labels = [
        label if label.endswith(".") else label + "." for label in labels
    ]

    results_detector: list[dict[str, Any]] = object_detector(
        image, candidate_labels=labels, threshold=threshold
    )
    results = [
        DetectionResult.from_dict(result) for result in results_detector
    ]

    return results


def segment(
    image: Image.Image,
    *,
    boxes: list[BoundingBox],
    segmentator: SAM2ImagePredictor,
    device: str,
) -> Image.Image:
    """Use Segment Anything Method (SAM) to generate a mask from an image and a.

        set of bounding boxes.

    Arguments:
        image (Image): The input image for which masks are to be generated.
        boxes (List[Tuple[int, int, int, int]]): A list of bounding
            boxes, where each box is represented as a tuple of (left, top,
            right, bottom) pixel coordinates.
        segmentator (SAM2ImagePredictor): The SAM2ImagePredictor model used
            for segmentation.
        device (str): The device on which the model should run (e.g., "cpu"
            or "cuda").

    Returns:
        Image: The output image with masks applied corresponding to the
            bounding boxes.

    Example:
        >>> segment_anything(image, [(50, 50, 100, 100), (150, 150, 200,
            200)], mask=initial_mask)

    Note:
        The function will raise an exception if any bounding box coordinates
            are out of the image boundaries.

    """
    bbox_xyxy = [list(box.xyxy) for box in boxes]
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        segmentator.set_image(image)
        # masks, _, _ = segmentator.predict(<input_prompts>)
        outputs: Bool[np.ndarray, "number_masks h w"] = segmentator.predict(
            box=np.asarray(bbox_xyxy)
        )[0].astype(bool)
    masks = torch.from_numpy(outputs)[None]
    if masks.shape[1] == 3:
        _masks = masks[:, 0].unsqueeze(1)
        for i in range(1, 3):
            _masks = torch.logical_or(_masks, masks[:, i].unsqueeze(1))
        masks = _masks
    return Image.fromarray((masks.cpu()[0, 0].numpy()))


@dataclass(config=ConfigDict(extra="forbid"))
class GroundingDino:
    """GroundingDino is a model that can ground text to image.

    It uses DINO for object detection and SAM for segmentation. Inspired from
    https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
    """

    dino_model_id: str = "IDEA-Research/grounding-dino-base"
    """ DINO model ID. """
    sam_model_id: Literal["facebook/sam2.1-hiera-large"] = (
        "facebook/sam2.1-hiera-large"
    )
    """ SAM model ID. """
    device: str = "cuda"
    """ Device to use. """

    def __post_init__(self) -> None:
        """Initialize the 'dino_model' and 'sam_model' attributes of the.

            GroundingDino instance.

        This method calls the 'init' method to initialize the 'dino_model'
            and 'sam_model' attributes of the GroundingDino instance.

        Arguments:
            self (GroundingDino instance): The instance of the GroundingDino
                class.

        Returns:
            None
        Example:
            >>> dino = GroundingDino()
            >>> dino.__init__()

        Note:
            This method is typically called when a new instance of
                GroundingDino is created.

        """
        self.dino_model, self.sam_model = self.init()

    def init(
        self,
    ) -> tuple[Pipeline, SAM2ImagePredictor]:
        """Initialize an object detector pipeline and a SAM2ImagePredictor.

            model.

        This method is part of the GroundingDino class and sets up the
            necessary pipelines
        and models for object detection and image prediction.

        Arguments:
            self (GroundingDino instance): The instance of the class
                GroundingDino.

        Returns:
            Tuple[ObjectDetectorPipeline, SAM2ImagePredictor]: A tuple
                containing the initialized object detector pipeline and
                SAM2ImagePredictor model.

        Example:
            >>> dino = GroundingDino()
            >>> pipeline, model = dino.initialize_pipeline()

        Note:
            This method should only be called once per instance of the
                GroundingDino class.

        """
        # build DINO
        object_detector = pipeline(
            model=self.dino_model_id,
            task="zero-shot-object-detection",
            device="cpu",
        )
        # build SAM
        sam_predictor = SAM2ImagePredictor.from_pretrained(
            self.sam_model_id
        )  # cant be cached
        sam_predictor.model = sam_predictor.model.to("cpu")
        return object_detector, sam_predictor

    def switch_to_cpu(self) -> None:
        """Switch the DINO and SAM models to CPU mode.

        This method changes the DINO and SAM models to operate on the CPU if
            the GPU is not currently in use. It is a part of the
            GroundingDino class.

        Arguments:
            self (GroundingDino instance): The instance of the GroundingDino
                class on which the method is invoked.

        Returns:
            None: This method does not return any value.

        Example:
            >>> dino = GroundingDino()
            >>> dino.switch_to_cpu()

        Note:
            This method is useful in scenarios where GPU resources are
                limited or not available.

        """
        logger.warning("Switching DINO to CPU")
        self.dino_model.device = torch.device("cpu")
        self.dino_model.model = self.dino_model.model.to("cpu")
        self.sam_model.model = self.sam_model.model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    def switch_to_cuda(self) -> None:
        """Switch the device of the Dino and SAM models to CUDA for GPU.

            acceleration.

        This method updates the device configuration of the Dino and SAM
            models to CUDA, allowing for GPU acceleration if a compatible
            GPU is available.

        Arguments:
            self: The instance of the GroundingDino class.

        Returns:
            None
        Example:
            >>> model = GroundingDino()
            >>> model.switch_to_cuda()

        Note:
            Ensure that a CUDA-compatible GPU is available for this method
                to have an effect. If no such GPU is available, the models
                will continue to use the default device.

        """
        self.dino_model.device = torch.device(self.device)
        self.dino_model.model = self.dino_model.model.to(self.device)
        self.sam_model.model = self.sam_model.model.to(self.device)

    @torch.inference_mode()
    def run(
        self,
        *,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.5,
        merge_masks: bool = False,
        verbose: bool = True,
    ) -> list[Image.Image]:
        """Perform object detection and segmentation on an input image based on.

            a given prompt.

        This method utilizes a DINO model for object detection and a SAM
            model for segmentation. It returns a list of segmented masks
            corresponding to the detected objects in the image.

        Arguments:
            image (HashableImage): The input image on which object detection
                and segmentation will be performed.
            prompt (str): The prompt or description of the objects to be
                detected in the image.
            number_of_objects (int): The number of objects to detect in the
                image.
            box_threshold (float, optional): The threshold for object
                detection bounding boxes. Defaults to 0.5.
            merge_masks (bool, optional): A flag indicating whether to merge
                the segmented masks. Defaults to False.
            verbose (bool, optional): A flag indicating whether to output
                debug information. Defaults to True.

        Returns:
            list[HashableImage]: A list of segmented masks corresponding to
                the detected objects in the input image.

        Example:
            >>> run(image, 'dog', 2, box_threshold=0.6, merge_masks=True,
                verbose=False)

        Note:
            This method is a part of the 'GroundingDino' class.

        """
        self.switch_to_cuda()
        sam_predictor = self.sam_model

        if not prompt.endswith("."):
            prompt += "."
        labels = prompt.lower().split(". ")
        detections = detect(
            image,
            labels=labels,
            threshold=box_threshold,
            object_detector=self.dino_model,
        )
        if verbose:
            logger.info(str(detections))
            logger.info(
                f"DINO number of objects {len(detections)} for {prompt=}"
            )
        list_mask = []
        if len(detections) > 0:
            for i in range(len(detections)):
                final_mask_hash = segment(
                    image,
                    boxes=[detections[i].box],
                    segmentator=sam_predictor,
                    device=self.device,
                )
                list_mask.append(final_mask_hash)

        if merge_masks:
            final_mask_np = np.zeros_like(list_mask[0])
            for mask in list_mask:
                final_mask_np = np.logical_or(final_mask_np, mask)
            list_mask = [Image.fromarray(final_mask_np)]
        return list_mask
