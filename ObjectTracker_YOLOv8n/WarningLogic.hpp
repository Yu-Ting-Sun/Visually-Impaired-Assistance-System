#ifndef WARNING_LOGIC_HPP
#define WARNING_LOGIC_HPP

#include "DetectionResult.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace arm
{
namespace app
{
namespace warning
{

enum class WarningSource
{
    None = 0,
    Tracker,
    RawFallback
};

enum class WarningSeverity
{
    None = 0,
    Safe,
    Caution,
    Danger
};

enum class DebugViewMode
{
    RawOnly = 0,
    RawWithTracks,
    WarningSummary
};

struct FrameObject
{
    int id = -1;
    WarningSource source = WarningSource::None;
    int class_id = -1;
    S_DETECTION_BOX bbox{};
    float score = 0.0f;
};

struct WarningEvent
{
    bool has_candidate = false;
    bool emitted = false;
    WarningSource source = WarningSource::None;
    WarningSeverity severity = WarningSeverity::None;
    int zone = 1;
    int class_id = -1;
    S_DETECTION_BOX bbox{};
    float path_overlap = 0.0f;
    float bottom_ratio = 0.0f;
    float height_ratio = 0.0f;
    float area_ratio = 0.0f;
    float approach_score = 0.0f;
    float risk_score = 0.0f;
};

inline const char *ToString(WarningSeverity severity)
{
    switch (severity)
    {
    case WarningSeverity::Safe:
        return "SAFE";
    case WarningSeverity::Caution:
        return "CAUTION";
    case WarningSeverity::Danger:
        return "DANGER";
    default:
        return "NONE";
    }
}

inline const char *ToString(WarningSource source)
{
    switch (source)
    {
    case WarningSource::Tracker:
        return "TRACK";
    case WarningSource::RawFallback:
        return "RAW";
    default:
        return "NONE";
    }
}

inline bool IsWarningClass(int cls)
{
    switch (cls)
    {
    case 0:  // person
    case 1:  // chair
    case 2:  // table
    case 4:  // couch
    case 24: // backpack
    case 26: // handbag
    case 28: // suitcase
        return true;
    default:
        return false;
    }
}

inline bool NeedsApproachEvidence(int cls)
{
    (void)cls;
    return false;
}

inline int ComputeZoneIndex(const S_DETECTION_BOX& bbox, int imgWidth)
{
    const float centerX = bbox.x + (bbox.w * 0.5f);
    if (centerX < imgWidth * 0.33f)
    {
        return 0;
    }
    if (centerX < imgWidth * 0.66f)
    {
        return 1;
    }
    return 2;
}

inline float ComputeCorridorHalfWidth(float imgWidth, float imgHeight, float y)
{
    const float topHalfWidth = imgWidth * 0.15f;
    const float bottomHalfWidth = imgWidth * 0.35f;
    const float normalizedY = std::clamp(y / std::max(1.0f, imgHeight - 1.0f), 0.0f, 1.0f);
    return topHalfWidth + ((bottomHalfWidth - topHalfWidth) * normalizedY);
}

inline float ComputeCorridorOverlapRatio(const S_DETECTION_BOX& bbox, int imgWidth, int imgHeight)
{
    if (bbox.w <= 1 || bbox.h <= 1)
    {
        return 0.0f;
    }

    const float left = static_cast<float>(bbox.x);
    const float top = static_cast<float>(bbox.y);
    const float right = left + bbox.w;
    const float bottom = top + bbox.h;
    const float corridorCenter = imgWidth * 0.5f;
    float overlapArea = 0.0f;

    for (int y = std::max(0, bbox.y); y < std::min(imgHeight, bbox.y + bbox.h); ++y)
    {
        const float sampleY = static_cast<float>(y) + 0.5f;
        const float halfWidth = ComputeCorridorHalfWidth(static_cast<float>(imgWidth), static_cast<float>(imgHeight), sampleY);
        const float corridorLeft = corridorCenter - halfWidth;
        const float corridorRight = corridorCenter + halfWidth;
        const float rowOverlap = std::max(0.0f, std::min(right, corridorRight) - std::max(left, corridorLeft));
        overlapArea += rowOverlap;
    }

    return overlapArea / (bbox.w * bbox.h);
}

inline float ComputeAreaRatio(const S_DETECTION_BOX& bbox, int imgWidth, int imgHeight)
{
    return static_cast<float>(bbox.w * bbox.h) / static_cast<float>(imgWidth * imgHeight);
}

inline float ComputeBottomRatio(const S_DETECTION_BOX& bbox, int imgHeight)
{
    return std::clamp(static_cast<float>(bbox.y + bbox.h) / static_cast<float>(imgHeight), 0.0f, 1.0f);
}

inline float ComputeHeightRatio(const S_DETECTION_BOX& bbox, int imgHeight)
{
    return std::clamp(static_cast<float>(bbox.h) / static_cast<float>(imgHeight), 0.0f, 1.0f);
}

inline float ComputeBoxIoU(const S_DETECTION_BOX& lhs, const S_DETECTION_BOX& rhs)
{
    const int left = std::max(lhs.x, rhs.x);
    const int top = std::max(lhs.y, rhs.y);
    const int right = std::min(lhs.x + lhs.w, rhs.x + rhs.w);
    const int bottom = std::min(lhs.y + lhs.h, rhs.y + rhs.h);

    const int overlapW = right - left;
    const int overlapH = bottom - top;
    if (overlapW <= 0 || overlapH <= 0)
    {
        return 0.0f;
    }

    const int intersection = overlapW * overlapH;
    const int unionArea = (lhs.w * lhs.h) + (rhs.w * rhs.h) - intersection;
    if (unionArea <= 0)
    {
        return 0.0f;
    }

    return static_cast<float>(intersection) / static_cast<float>(unionArea);
}

class WarningEngine
{
public:
    WarningEngine(int imgWidth, int imgHeight)
        : m_imgWidth(imgWidth),
          m_imgHeight(imgHeight)
    {
    }

    void Reset()
    {
        m_hasPrevCandidate = false;
        m_stableFrames = 0;
        m_clearFrames = 0;
        m_lastEmitFrame = 0;
        m_lastEmittedSeverity = WarningSeverity::None;
        m_prevCandidate = CandidateState{};
    }

    WarningEvent Update(
        uint32_t frameSeq,
        const std::vector<FrameObject>& primaryObjects,
        const std::vector<FrameObject>& fallbackObjects)
    {
        CandidateState candidate = SelectBestCandidate(primaryObjects);
        if (!candidate.valid)
        {
            candidate = SelectBestCandidate(fallbackObjects);
        }

        return CommitFrame(frameSeq, candidate);
    }

    WarningEvent Update(uint32_t frameSeq, const std::vector<FrameObject>& objects)
    {
        return CommitFrame(frameSeq, SelectBestCandidate(objects));
    }

private:
    struct CandidateState
    {
        bool valid = false;
        int id = -1;
        WarningSource source = WarningSource::None;
        int classId = -1;
        int zone = 1;
        S_DETECTION_BOX bbox{};
        float score = 0.0f;
        float pathOverlap = 0.0f;
        float bottomRatio = 0.0f;
        float heightRatio = 0.0f;
        float areaRatio = 0.0f;
        float approachScore = 0.0f;
        float riskScore = 0.0f;
        WarningSeverity severity = WarningSeverity::None;
    };

    CandidateState EvaluateObject(const FrameObject& object) const
    {
        CandidateState candidate{};

        if (!IsWarningClass(object.class_id) || object.bbox.w <= 1 || object.bbox.h <= 1)
        {
            return candidate;
        }

        candidate.valid = true;
        candidate.id = object.id;
        candidate.source = object.source;
        candidate.classId = object.class_id;
        candidate.zone = ComputeZoneIndex(object.bbox, m_imgWidth);
        candidate.bbox = object.bbox;
        candidate.score = object.score;
        candidate.pathOverlap = ComputeCorridorOverlapRatio(object.bbox, m_imgWidth, m_imgHeight);
        candidate.bottomRatio = ComputeBottomRatio(object.bbox, m_imgHeight);
        candidate.heightRatio = ComputeHeightRatio(object.bbox, m_imgHeight);
        candidate.areaRatio = ComputeAreaRatio(object.bbox, m_imgWidth, m_imgHeight);

        if (candidate.pathOverlap < 0.35f)
        {
            candidate.valid = false;
            return candidate;
        }

        candidate.riskScore = ComputeRiskScore(candidate);
        candidate.severity = ComputeSeverity(candidate);
        return candidate;
    }

    CandidateState SelectBestCandidate(const std::vector<FrameObject>& objects) const
    {
        CandidateState candidate{};

        for (const auto& object : objects)
        {
            CandidateState current = EvaluateObject(object);
            if (!current.valid)
            {
                continue;
            }

            if (!candidate.valid || IsBetterCandidate(current, candidate))
            {
                candidate = current;
            }
        }

        return candidate;
    }

    WarningEvent CommitFrame(uint32_t frameSeq, CandidateState candidate)
    {
        WarningEvent event{};

        if (!candidate.valid)
        {
            m_clearFrames = static_cast<uint8_t>(std::min(255, m_clearFrames + 1));
            if (m_clearFrames >= 3)
            {
                m_stableFrames = 0;
                m_hasPrevCandidate = false;
                m_prevCandidate = CandidateState{};
                m_lastEmittedSeverity = WarningSeverity::None;
            }
            return event;
        }

        m_clearFrames = 0;

        if (m_hasPrevCandidate && MatchesPrevious(candidate, m_prevCandidate))
        {
            m_stableFrames = static_cast<uint8_t>(std::min(255, m_stableFrames + 1));
            candidate.approachScore = ComputeApproachScore(candidate, m_prevCandidate);
            candidate.riskScore = ComputeRiskScore(candidate);
            candidate.severity = ComputeSeverity(candidate);
        }
        else
        {
            m_stableFrames = 1;
        }

        m_prevCandidate = candidate;
        m_hasPrevCandidate = true;

        event.has_candidate = true;
        event.source = candidate.source;
        event.severity = candidate.severity;
        event.zone = candidate.zone;
        event.class_id = candidate.classId;
        event.bbox = candidate.bbox;
        event.path_overlap = candidate.pathOverlap;
        event.bottom_ratio = candidate.bottomRatio;
        event.height_ratio = candidate.heightRatio;
        event.area_ratio = candidate.areaRatio;
        event.approach_score = candidate.approachScore;
        event.risk_score = candidate.riskScore;

        if (ShouldEmit(frameSeq, candidate))
        {
            event.emitted = true;
            m_lastEmitFrame = frameSeq;
            m_lastEmittedSeverity = candidate.severity;
        }

        return event;
    }

    float ComputeApproachScore(const CandidateState& current, const CandidateState& previous) const
    {
        const float bottomDelta = std::max(0.0f, current.bottomRatio - previous.bottomRatio);
        const float heightDelta = std::max(0.0f, current.heightRatio - previous.heightRatio);
        return bottomDelta + (heightDelta * 0.5f);
    }

    float ComputeRiskScore(const CandidateState& candidate) const
    {
        const float areaBoost = std::clamp(candidate.areaRatio / 0.20f, 0.0f, 1.0f);
        const float sizeScore = std::clamp((candidate.heightRatio * 0.7f) + (areaBoost * 0.3f), 0.0f, 1.0f);
        const float trendScore = std::clamp(candidate.approachScore * 8.0f, 0.0f, 1.0f);

        return (candidate.pathOverlap * 0.40f) +
               (candidate.bottomRatio * 0.30f) +
               (sizeScore * 0.20f) +
               (trendScore * 0.10f);
    }

    WarningSeverity ComputeSeverity(const CandidateState& candidate) const
    {
        if (NeedsApproachEvidence(candidate.classId) && candidate.approachScore < 0.03f)
        {
            return WarningSeverity::Safe;
        }

        if (candidate.riskScore >= 0.82f ||
            (candidate.bottomRatio >= 0.92f && candidate.heightRatio >= 0.55f && candidate.pathOverlap >= 0.5f))
        {
            return WarningSeverity::Danger;
        }

        if (candidate.riskScore >= 0.62f)
        {
            return WarningSeverity::Caution;
        }

        return WarningSeverity::Safe;
    }

    bool IsBetterCandidate(const CandidateState& lhs, const CandidateState& rhs) const
    {
        if (lhs.riskScore != rhs.riskScore)
        {
            return lhs.riskScore > rhs.riskScore;
        }

        if (lhs.source != rhs.source)
        {
            return lhs.source == WarningSource::Tracker;
        }

        return lhs.score > rhs.score;
    }

    bool MatchesPrevious(const CandidateState& current, const CandidateState& previous) const
    {
        if (!previous.valid || current.source != previous.source || current.classId != previous.classId)
        {
            return false;
        }

        if (current.source == WarningSource::Tracker)
        {
            return current.id == previous.id;
        }

        return ComputeBoxIoU(current.bbox, previous.bbox) >= 0.5f;
    }

    bool ShouldEmit(uint32_t frameSeq, const CandidateState& candidate) const
    {
        if (candidate.severity == WarningSeverity::Safe || candidate.severity == WarningSeverity::None)
        {
            return false;
        }

        const uint8_t requiredStableFrames =
            (candidate.source == WarningSource::Tracker)
                ? (candidate.severity == WarningSeverity::Danger ? 2 : 3)
                : (candidate.severity == WarningSeverity::Danger ? 3 : 4);

        if (m_stableFrames < requiredStableFrames)
        {
            return false;
        }

        const uint32_t cooldownFrames = (candidate.severity == WarningSeverity::Danger) ? 8U : 14U;
        const bool sameSeverity = candidate.severity == m_lastEmittedSeverity;
        const bool underCooldown = sameSeverity && ((frameSeq - m_lastEmitFrame) < cooldownFrames);
        return !underCooldown;
    }

    int m_imgWidth = 0;
    int m_imgHeight = 0;
    bool m_hasPrevCandidate = false;
    uint8_t m_stableFrames = 0;
    uint8_t m_clearFrames = 0;
    uint32_t m_lastEmitFrame = 0;
    WarningSeverity m_lastEmittedSeverity = WarningSeverity::None;
    CandidateState m_prevCandidate{};
};

} /* namespace warning */
} /* namespace app */
} /* namespace arm */

#endif /* WARNING_LOGIC_HPP */
