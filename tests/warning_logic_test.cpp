#include "WarningLogic.hpp"

#include <cstdlib>
#include <iostream>

using namespace arm::app::warning;

namespace
{

void Expect(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

FrameObject MakeObject(
    int id,
    WarningSource source,
    int cls,
    int x,
    int y,
    int w,
    int h,
    float score)
{
    FrameObject object{};
    object.id = id;
    object.source = source;
    object.class_id = cls;
    object.bbox = {x, y, w, h, cls, score};
    object.score = score;
    return object;
}

void TestCorridorOverlapPrefersForwardObjects()
{
    const float forwardOverlap = ComputeCorridorOverlapRatio({110, 100, 100, 120, 0, 0.9f}, 320, 240);
    const float sideOverlap = ComputeCorridorOverlapRatio({0, 100, 60, 120, 0, 0.9f}, 320, 240);

    Expect(forwardOverlap > 0.35f, "Forward object should overlap the navigation corridor");
    Expect(sideOverlap < 0.35f, "Side object should stay outside the navigation corridor");
}

void TestWarningWhitelistRejectsSmallPortableClasses()
{
    Expect(IsWarningClass(0), "person should be in the warning whitelist");
    Expect(IsWarningClass(1), "chair should be in the warning whitelist");
    Expect(IsWarningClass(2), "table should be in the warning whitelist");
    Expect(IsWarningClass(4), "couch should be in the warning whitelist");
    Expect(IsWarningClass(24), "backpack should be in the warning whitelist");
    Expect(IsWarningClass(26), "handbag should be in the warning whitelist");
    Expect(IsWarningClass(28), "suitcase should be in the warning whitelist");
    Expect(!IsWarningClass(15), "bottle should stay out of the warning whitelist");
    Expect(!NeedsApproachEvidence(2), "indoor warning classes should not require class-specific approach evidence");
}

void TestDangerFastPathTriggersOnRapidApproach()
{
    WarningEngine engine(320, 240);
    WarningEvent event{};

    const FrameObject trackerCandidate = MakeObject(7, WarningSource::Tracker, 0, 110, 80, 110, 150, 0.92f);

    event = engine.Update(1, {trackerCandidate});
    Expect(!event.emitted, "Single frame should not immediately emit danger");

    FrameObject closer = trackerCandidate;
    closer.bbox.y = 70;
    closer.bbox.h = 170;
    event = engine.Update(2, {closer});
    Expect(event.emitted, "Danger fast path should emit after consecutive rapid approach frames");
    Expect(event.severity == WarningSeverity::Danger, "Danger fast path should emit DANGER");
}

void TestRawFallbackNeedsStableFrames()
{
    WarningEngine engine(320, 240);
    WarningEvent event{};

    FrameObject rawCandidate = MakeObject(-1, WarningSource::RawFallback, 1, 110, 90, 100, 130, 0.88f);

    event = engine.Update(1, {rawCandidate});
    Expect(!event.emitted, "Raw fallback should not emit on the first frame");

    rawCandidate.bbox.y = 86;
    rawCandidate.bbox.h = 136;
    event = engine.Update(2, {rawCandidate});
    Expect(!event.emitted, "Raw fallback should require more than two stable frames");

    rawCandidate.bbox.y = 82;
    rawCandidate.bbox.h = 144;
    event = engine.Update(3, {rawCandidate});
    Expect(event.emitted, "Raw fallback should emit after stable consecutive frames");
    Expect(event.source == WarningSource::RawFallback, "Raw fallback event should report its source");
}

} // namespace

int main()
{
    TestCorridorOverlapPrefersForwardObjects();
    TestWarningWhitelistRejectsSmallPortableClasses();
    TestDangerFastPathTriggersOnRapidApproach();
    TestRawFallbackNeedsStableFrames();
    std::cout << "warning_logic_test: PASS\n";
    return 0;
}
