#ifndef YOLOV8N_OD_POST_PROCESSING_HPP
#define YOLOV8N_OD_POST_PROCESSING_HPP

#include "PostProcessUtils.hpp"
#include "YOLOv8nODModel.hpp"

#include <forward_list>

namespace arm
{
namespace app
{
namespace yolov8n_od
{

#define YOLOV8N_OD_STRIDE_8	(8)
#define YOLOV8N_OD_STRIDE_16	(16)
#define YOLOV8N_OD_STRIDE_32	(32)

	
/**
 * @brief   Helper class to manage tensor post-processing for "yolov8n pose"
 *          output.
 */
class YOLOv8nODPostProcessing
{
public:
    /**
     * @brief       Constructor.
     * @param[in]   threshold     Post-processing threshold.
     **/
    explicit YOLOv8nODPostProcessing(arm::app::YOLOv8nODModel *model, float threshold = 0.5f);

    /**
     * @brief       Post processing part of YOLOv8n pose model.
     * @param[in]   imgNetRows      Number of rows in the network input image.
     * @param[in]   imgNetCols      Number of columns in the network input image.
     * @param[in]   imgSrcRows      Number of rows in the orignal input image.
     * @param[in]   imgSrcCols      Number of columns in the oringal input image.
     * @param[out]  resultsOut   Vector of detected results.
     **/
    void RunPostProcessing(uint32_t imgNetRows,
                           uint32_t imgNetCols,
                           uint32_t imgSrcRows,
                           uint32_t imgSrcCols,
                           std::vector<DetectionResult> &resultsOut);

private:
	arm::app::YOLOv8nODModel *m_model;
    float m_threshold;  /* Post-processing threshold */
	int m_stride8_total_anchors;		//total anchors for stride 8
	int m_stride16_total_anchors;		//total anchors for stride 16
	int m_stride32_total_anchors;		//total anchors for stride 32

	std::vector<AnchorBox> m_stride8_anchros;
	std::vector<AnchorBox> m_stride16_anchros;
	std::vector<AnchorBox> m_stride32_anchros;
    OutputTensorMapping m_outputTensorMapping;

	void GetNetworkBoxes(std::forward_list<Detection>& detections);
    bool ResolveOutputTensorMapping();

};

} /* namespace yolov8n_od */
} /* namespace app */
} /* namespace arm */

#endif /* YOLOV8N_OD_POST_PROCESSING_HPP */
