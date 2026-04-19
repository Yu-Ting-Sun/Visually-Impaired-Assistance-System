#include "BufAttributes.hpp"

#include <string>
#include <vector>

static const char *labelsVec[] LABELS_ATTRIBUTE =
{
    "person",
    "chair",
    "table",
    "desk",
    "couch",
    "bed",
    "cabinetry",
    "shelf",
    "countertop",
    "door",
    "window",
    "laptop",
    "mobile phone",
    "computer monitor",
    "television",
    "bottle",
    "coffee cup",
    "mug",
    "bowl",
    "plate",
    "book",
    "bench",
    "bookcase",
    "drawer",
    "backpack",
    "closet",
    "handbag",
    "toilet",
    "suitcase",
    "mirror",
    "houseplant",
    "lamp",
    "picture frame",
    "tablet computer",
    "computer keyboard",
    "computer mouse",
    "remote control",
    "headphones",
    "printer",
    "microwave oven",
    "oven",
    "toaster",
    "coffeemaker",
    "blender",
    "washing machine",
    "fork",
    "spoon",
    "kitchen knife",
    "clock",
    "scissors",
};

bool GetLabelsVector(std::vector<std::string> &labels)
{
    constexpr size_t labelsSz = sizeof(labelsVec) / sizeof(labelsVec[0]);
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
