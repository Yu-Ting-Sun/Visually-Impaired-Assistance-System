
#include "BufAttributes.hpp"

#include <vector>
#include <string>

static const char *labelsVec[] LABELS_ATTRIBUTE =
{
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    /* 80-85 from taiwan_street */
    "crossing_crosswalk",
    "curb",
    "green_sidewalk",
    "guide_bricks",
    "sidewalk",
    "stairs",
    /* 86-89 from pedestrian_lights */
    "green",
    "pedestrian_traffic_light",
    "red",
    "signal-light",
    /* 90-92 from braille_blocks */
    "braille_blocks",
    "dots",
    "lines",
};

bool GetLabelsVector(std::vector<std::string> &labels)
{
    constexpr size_t labelsSz = 93;
    labels.clear();

    if (!labelsSz)
    {
        return false;
    }

    labels.reserve(labelsSz);

    for (size_t i = 0; i < labelsSz; ++i)
    {
        labels.emplace_back(labelsVec[i]);
    }

    return true;
}

