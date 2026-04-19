#ifndef POST_PROCESS_UTILS_HPP
#define POST_PROCESS_UTILS_HPP

#include "DetectionResult.hpp"

#include <algorithm>
#include <cmath>
#include <forward_list>
#include <vector>

namespace arm
{
namespace app
{
namespace yolov8n_od
{

struct Box {
    float x;
    float y;
    float w;
    float h;
};

struct Detection {
    Box bbox;
    int strideIndex;
    int anchorIndex;
    int cls;
    std::vector<float> prob;
};

struct AnchorBox {
    float w;
    float h;
};

struct OutputTensorShape {
    int index;
    int dimsSize;
    int anchors;
    int valuesPerAnchor;
};

struct OutputTensorMapping {
    bool valid = false;
    int stride8Box = -1;
    int stride8Confidence = -1;
    int stride16Box = -1;
    int stride16Confidence = -1;
    int stride32Box = -1;
    int stride32Confidence = -1;
};

inline float CalculateBoxIntersect(const Box& box1, const Box& box2)
{
    const float left = std::max(box1.x, box2.x);
    const float top = std::max(box1.y, box2.y);
    const float right = std::min(box1.x + box1.w, box2.x + box2.w);
    const float bottom = std::min(box1.y + box1.h, box2.y + box2.h);

    const float overlapW = right - left;
    const float overlapH = bottom - top;

    if (overlapW <= 0.0f || overlapH <= 0.0f)
    {
        return 0.0f;
    }

    return overlapW * overlapH;
}

inline float CalculateBoxUnion(const Box& box1, const Box& box2)
{
    const float intersection = CalculateBoxIntersect(box1, box2);
    return (box1.w * box1.h) + (box2.w * box2.h) - intersection;
}

inline float CalculateBoxIOU(const Box& box1, const Box& box2)
{
    const float intersection = CalculateBoxIntersect(box1, box2);
    if (intersection <= 0.0f)
    {
        return 0.0f;
    }

    const float unionArea = CalculateBoxUnion(box1, box2);
    if (unionArea <= 0.0f)
    {
        return 0.0f;
    }

    return intersection / unionArea;
}

inline bool ValidateBoxTensorShape(int anchors, int boxDataSize, int expectedAnchors)
{
    return anchors == expectedAnchors && boxDataSize == 64;
}

inline bool ValidateConfidenceTensorShape(
    int dimsSize,
    int anchors,
    int classes,
    int expectedAnchors,
    int expectedClasses)
{
    return dimsSize == 3 && anchors == expectedAnchors && classes == expectedClasses;
}

inline void CalculateNMS(std::forward_list<Detection>& detections, int classes, float iouThreshold)
{
    for (int idxClass = 0; idxClass < classes; ++idxClass)
    {
        detections.sort([idxClass](const Detection& lhs, const Detection& rhs) {
            return lhs.prob[idxClass] > rhs.prob[idxClass];
        });

        for (auto it = detections.begin(); it != detections.end(); ++it)
        {
            if (it->prob[idxClass] <= 0.0f)
            {
                continue;
            }

            for (auto itc = std::next(it, 1); itc != detections.end(); ++itc)
            {
                if (itc->prob[idxClass] <= 0.0f)
                {
                    continue;
                }

                if (CalculateBoxIOU(it->bbox, itc->bbox) > iouThreshold)
                {
                    itc->prob[idxClass] = 0.0f;
                }
            }
        }
    }
}

inline S_DETECTION_BOX ClampAndValidateDetectionBox(
    float x,
    float y,
    float w,
    float h,
    int cls,
    float score,
    int imgSrcCols,
    int imgSrcRows)
{
    S_DETECTION_BOX invalid{};
    invalid.cls = cls;
    invalid.normalisedVal = score;

    if (w <= 1.0f || h <= 1.0f)
    {
        return invalid;
    }

    float left = x;
    float top = y;
    float right = x + w;
    float bottom = y + h;

    if (right <= 0.0f || bottom <= 0.0f || left >= imgSrcCols || top >= imgSrcRows)
    {
        return invalid;
    }

    left = std::max(left, 0.0f);
    top = std::max(top, 0.0f);
    right = std::min(right, static_cast<float>(imgSrcCols));
    bottom = std::min(bottom, static_cast<float>(imgSrcRows));

    const int clampedX = static_cast<int>(std::floor(left));
    const int clampedY = static_cast<int>(std::floor(top));
    const int clampedW = static_cast<int>(std::floor(right - left));
    const int clampedH = static_cast<int>(std::floor(bottom - top));

    if (clampedW <= 1 || clampedH <= 1)
    {
        return invalid;
    }

    S_DETECTION_BOX box{};
    box.x = clampedX;
    box.y = clampedY;
    box.w = clampedW;
    box.h = clampedH;
    box.cls = cls;
    box.normalisedVal = score;
    return box;
}

inline bool IsValidDetectionBox(const S_DETECTION_BOX& box)
{
    return box.w > 1 && box.h > 1;
}

inline bool AssignOutputTensorIndex(OutputTensorMapping& mapping, int anchors, bool isBoxTensor, int index, int stride8Anchors, int stride16Anchors, int stride32Anchors)
{
    int* target = nullptr;

    if (anchors == stride8Anchors)
    {
        target = isBoxTensor ? &mapping.stride8Box : &mapping.stride8Confidence;
    }
    else if (anchors == stride16Anchors)
    {
        target = isBoxTensor ? &mapping.stride16Box : &mapping.stride16Confidence;
    }
    else if (anchors == stride32Anchors)
    {
        target = isBoxTensor ? &mapping.stride32Box : &mapping.stride32Confidence;
    }
    else
    {
        return false;
    }

    if (*target != -1)
    {
        return false;
    }

    *target = index;
    return true;
}

inline OutputTensorMapping ResolveOutputTensorMapping(
    const std::vector<OutputTensorShape>& shapes,
    int expectedClasses,
    int stride8Anchors,
    int stride16Anchors,
    int stride32Anchors)
{
    OutputTensorMapping mapping{};

    for (const auto& shape : shapes)
    {
        if (shape.dimsSize != 3)
        {
            continue;
        }

        if (shape.valuesPerAnchor == 64)
        {
            if (!AssignOutputTensorIndex(mapping, shape.anchors, true, shape.index, stride8Anchors, stride16Anchors, stride32Anchors))
            {
                return OutputTensorMapping{};
            }
            continue;
        }

        if (shape.valuesPerAnchor == expectedClasses)
        {
            if (!AssignOutputTensorIndex(mapping, shape.anchors, false, shape.index, stride8Anchors, stride16Anchors, stride32Anchors))
            {
                return OutputTensorMapping{};
            }
        }
    }

    mapping.valid =
        mapping.stride8Box >= 0 &&
        mapping.stride8Confidence >= 0 &&
        mapping.stride16Box >= 0 &&
        mapping.stride16Confidence >= 0 &&
        mapping.stride32Box >= 0 &&
        mapping.stride32Confidence >= 0;

    return mapping;
}

} /* namespace yolov8n_od */
} /* namespace app */
} /* namespace arm */

#endif /* POST_PROCESS_UTILS_HPP */
