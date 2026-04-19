#include "PostProcessUtils.hpp"

#include <cmath>
#include <cstdlib>
#include <forward_list>
#include <iostream>

using namespace arm::app::yolov8n_od;

namespace
{

bool NearlyEqual(float lhs, float rhs, float eps = 1e-5f)
{
    return std::fabs(lhs - rhs) <= eps;
}

void Expect(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void TestIoUUsesTopLeftGeometry()
{
    const Box boxA{10.0f, 10.0f, 20.0f, 20.0f};
    const Box boxB{20.0f, 20.0f, 20.0f, 20.0f};

    const float intersection = 10.0f * 10.0f;
    const float unionArea = (20.0f * 20.0f) + (20.0f * 20.0f) - intersection;
    const float expectedIoU = intersection / unionArea;

    Expect(NearlyEqual(CalculateBoxIOU(boxA, boxB), expectedIoU),
           "IoU should treat x/y as top-left coordinates");
}

void TestClasswiseNmsSortsByEachClass()
{
    Detection detA{};
    detA.bbox = {0.0f, 0.0f, 20.0f, 20.0f};
    detA.cls = 1;
    detA.prob = {0.95f, 0.10f};

    Detection detB{};
    detB.bbox = {0.0f, 0.0f, 20.0f, 20.0f};
    detB.cls = 1;
    detB.prob = {0.20f, 0.90f};

    std::forward_list<Detection> detections;
    detections.push_front(detA);
    detections.push_front(detB);

    CalculateNMS(detections, 2, 0.5f);

    int survivorsForClass1 = 0;
    for (const auto &det : detections)
    {
        if (det.prob[1] > 0.0f)
        {
            ++survivorsForClass1;
            Expect(NearlyEqual(det.prob[1], 0.90f),
                   "NMS should keep the highest score for the active class");
        }
    }

    Expect(survivorsForClass1 == 1, "NMS should suppress duplicate boxes per class");
}

void TestClampConvertedBoxRejectsDegenerateBoxes()
{
    const S_DETECTION_BOX valid = ClampAndValidateDetectionBox(10.0f, 15.0f, 40.0f, 50.0f, 3, 0.8f, 320, 240);
    Expect(valid.w == 40 && valid.h == 50, "Valid boxes should stay unchanged");

    const S_DETECTION_BOX invalid = ClampAndValidateDetectionBox(100.0f, 100.0f, 0.0f, 20.0f, 2, 0.7f, 320, 240);
    Expect(invalid.w == 0 && invalid.h == 0,
           "Degenerate boxes should be converted into invalid zero-sized boxes");
}

void TestResolveOutputTensorLayoutByShape()
{
    const std::vector<OutputTensorShape> shapes = {
        {0, 3, 144, 50},
        {1, 3, 576, 64},
        {2, 3, 36, 50},
        {3, 3, 576, 50},
        {4, 3, 36, 64},
        {5, 3, 144, 64},
    };

    const OutputTensorMapping mapping = ResolveOutputTensorMapping(shapes, 50, 576, 144, 36);

    Expect(mapping.valid, "Output tensor mapping should succeed for shuffled 6-output YOLO heads");
    Expect(mapping.stride8Box == 1, "stride-8 box tensor should be resolved by shape");
    Expect(mapping.stride8Confidence == 3, "stride-8 confidence tensor should be resolved by shape");
    Expect(mapping.stride16Box == 5, "stride-16 box tensor should be resolved by shape");
    Expect(mapping.stride16Confidence == 0, "stride-16 confidence tensor should be resolved by shape");
    Expect(mapping.stride32Box == 4, "stride-32 box tensor should be resolved by shape");
    Expect(mapping.stride32Confidence == 2, "stride-32 confidence tensor should be resolved by shape");
}

} // namespace

int main()
{
    TestIoUUsesTopLeftGeometry();
    TestClasswiseNmsSortsByEachClass();
    TestClampConvertedBoxRejectsDegenerateBoxes();
    TestResolveOutputTensorLayoutByShape();
    std::cout << "postprocess_utils_test: PASS\n";
    return 0;
}
