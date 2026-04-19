#include "YOLOv8nODPostProcessing.hpp"
#include "PlatformMath.hpp"

#include <cmath>

using namespace arm::app::yolov8n_od;

/************** YOLOv8n-pose  */ 

#if (YOLOV8_OD_INPUT_TENSOR == 224)
#define YOLOV8_OD_INPUT_TENSOR_WIDTH   224
#define YOLOV8_OD_INPUT_TENSOR_HEIGHT  224
#elif (YOLOV8_OD_INPUT_TENSOR == 256)
#define YOLOV8_OD_INPUT_TENSOR_WIDTH   256
#define YOLOV8_OD_INPUT_TENSOR_HEIGHT  256
#else		//192
#define YOLOV8_OD_INPUT_TENSOR_WIDTH   192
#define YOLOV8_OD_INPUT_TENSOR_HEIGHT  192
#endif

void AnchorMatrixConstruct(
	std::vector<AnchorBox> &vAnchorBoxs,
	int i32Stride,
	int i32StrideTotalAnchors
)
{
	int i;
	float fStartAnchorValue = 0.5;
	int iMaxAnchorValue = (YOLOV8_OD_INPUT_TENSOR_WIDTH/i32Stride);
	float fAnchor0StepValue = 0.;
	float fAnchor1StepValue = -1.;

	for(int i = 0; i < i32StrideTotalAnchors; i++)
	{
		AnchorBox sAnchorBox;

		if((i % iMaxAnchorValue)==0)
		{
			fStartAnchorValue = 0.5;
			fAnchor0StepValue = 0.;
			fAnchor1StepValue++;
		}

		sAnchorBox.w = fStartAnchorValue + (fAnchor0StepValue++);
		sAnchorBox.h = fStartAnchorValue + fAnchor1StepValue;
		
		vAnchorBoxs.push_back(sAnchorBox);
	}

	//for(int i=0; i < vAnchorBoxs.size(); i++)
	//{
	//	printf("vAnchorBoxs[%d].w = %f \n", i, vAnchorBoxs[i].w);
	//	printf("vAnchorBoxs[%d].h = %f \n", i, vAnchorBoxs[i].h);		
	//}
}

void CalBoxXYWH(
	TfLiteTensor* psBoxOutputTensor,
	std::vector<AnchorBox> &vAnchorBoxs,
	int	i32AnchorIndex,
	int i32Stride,
	int i32StrideTotalAnchors,
	Detection &sDetection
)
{
	int i;
    float scaleBox;
    int zeroPointBox;
	int anchors;
	int boxDataSize;
    float  XYWHResult[4];
	
	int8_t *tensorOutputBox = psBoxOutputTensor->data.int8;
    scaleBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->scale->data[0];
    zeroPointBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->zero_point->data[0];

	anchors = psBoxOutputTensor->dims->data[1];
	boxDataSize = psBoxOutputTensor->dims->data[2];

	if (!ValidateBoxTensorShape(anchors, boxDataSize, i32StrideTotalAnchors))
	{
		printf("CalBoxXYWH(): error tensor size not match \n");
		return;
	}

	tensorOutputBox = tensorOutputBox + (i32AnchorIndex * boxDataSize);
	
    for(int k = 0 ; k < 4 ; k++)
    {
		std::vector<float> XYWHSoftmaxTemp(16);
        float XYWHSoftmaxResult=0;

        for(int i = 0 ; i < 16 ; i++)
        {
			XYWHSoftmaxTemp[i] = scaleBox * (static_cast<float>(tensorOutputBox[k*16 + i]) - zeroPointBox);
		}

		arm::app::math::MathUtils::SoftmaxF32(XYWHSoftmaxTemp);
        for(int i = 0 ; i < 16 ; i++)
        {

            XYWHSoftmaxResult = XYWHSoftmaxResult + XYWHSoftmaxTemp[i]*i;
        }
        XYWHResult[k] = XYWHSoftmaxResult;
	}

    /* dist2bbox */
    float x1 = vAnchorBoxs[i32AnchorIndex].w -  XYWHResult[0];
    float y1 = vAnchorBoxs[i32AnchorIndex].h -  XYWHResult[1];
    float x2 = vAnchorBoxs[i32AnchorIndex].w +  XYWHResult[2];
    float y2 = vAnchorBoxs[i32AnchorIndex].h +  XYWHResult[3];
    
    float cx = (x1 + x2)/2.;
    float cy = (y1 + y2)/2.;
    float w = x2 - x1;
    float h = y2 - y1;

    XYWHResult[0] = cx * i32Stride;
    XYWHResult[1] = cy * i32Stride;
    XYWHResult[2] = w * i32Stride;
    XYWHResult[3] = h * i32Stride;

	sDetection.bbox.x = XYWHResult[0] - (0.5 * XYWHResult[2]);
    sDetection.bbox.y = XYWHResult[1] - (0.5 * XYWHResult[3]);
	sDetection.bbox.w = XYWHResult[2];
    sDetection.bbox.h = XYWHResult[3];
}

void CalDetectionBox(
	TfLiteTensor* psConfidenceOutputTensor,
	TfLiteTensor* psBoxOutputTensor,
	std::vector<AnchorBox> &vAnchorBoxs,
	int i32Stride,
	int i32StrideTotalAnchors,
	float fThreshold,
	std::forward_list<Detection>&sDetections		
)
{
	int i, j;
    float scaleConf;
    int zeroPointConf;
	float maxScore = 0.0;
	int maxConf;
	int cls= 0;
	int8_t *tensorOutputConf = psConfidenceOutputTensor->data.int8;

    scaleConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->scale->data[0];
    zeroPointConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->zero_point->data[0];

	if (!ValidateConfidenceTensorShape(
			psConfidenceOutputTensor->dims->size,
			psConfidenceOutputTensor->dims->data[1],
			psConfidenceOutputTensor->dims->data[2],
			i32StrideTotalAnchors,
			YOLOV8N_OD_CLASS))
	{
		printf("CalDetectionBox(): error tensor size not match \n");
		return;
	}

	//check confidence is over threshold or not
	for(i = 0 ; i <i32StrideTotalAnchors; i++)
	{
		maxScore = 0.0;
		cls = 0;
		maxConf = -128;

		for(j = 0; j < YOLOV8N_OD_CLASS; j ++)
		{
			int confTesorData;
			
			confTesorData  = tensorOutputConf[(i * YOLOV8N_OD_CLASS) + j];
			
			if(confTesorData > maxConf)
			{
				maxConf = confTesorData;
				cls = j;				
			}
		}		

		maxScore = arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(maxConf - zeroPointConf)));		
		
		if(maxScore >= fThreshold)
		{
			//printf("max cls %d and score %f i32StrideTotalAnchors %d, index %d\n", cls, maxScore, i32StrideTotalAnchors, i);
			//inqueue detection list
			arm::app::yolov8n_od::Detection det;
			det.strideIndex = i32Stride;
			det.anchorIndex = i;
			det.cls = cls;

			for(j = 0; j < YOLOV8N_OD_CLASS; j ++){
				maxScore =  arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(tensorOutputConf[(i * YOLOV8N_OD_CLASS) + j] - zeroPointConf)));
				det.prob.emplace_back(maxScore);
			}

			//cal box xywh
			CalBoxXYWH(psBoxOutputTensor,
				vAnchorBoxs,
				i,
				i32Stride,
				i32StrideTotalAnchors,
				det);
            sDetections.emplace_front(det);
		}
	}
}


/*****************************/
namespace arm
{
namespace app
{
namespace yolov8n_od
{

YOLOv8nODPostProcessing::YOLOv8nODPostProcessing(
	arm::app::YOLOv8nODModel *model,
	const float threshold)
    :   m_threshold(threshold),
		m_model(model)
{
	int i;

	//For YOLOV8_OD_INPUT_TENSOR == 256, it would be 1024
	//For YOLOV8_OD_INPUT_TENSOR == 192, it would be 576	
	m_stride8_total_anchors = pow(( YOLOV8_OD_INPUT_TENSOR_WIDTH / YOLOV8N_OD_STRIDE_8),2);
	//For YOLOV8_OD_INPUT_TENSOR == 256, it would be 256
	//For YOLOV8_OD_INPUT_TENSOR == 192, it would be 144	
	m_stride16_total_anchors = pow(( YOLOV8_OD_INPUT_TENSOR_WIDTH / YOLOV8N_OD_STRIDE_16),2);
	//For YOLOV8_OD_INPUT_TENSOR == 256, it would be 64
	//For YOLOV8_OD_INPUT_TENSOR == 192, it would be 36	
	m_stride32_total_anchors = pow(( YOLOV8_OD_INPUT_TENSOR_WIDTH / YOLOV8N_OD_STRIDE_32),2);

	m_stride8_anchros.clear();
	m_stride16_anchros.clear();
	m_stride32_anchros.clear();
	
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[31.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride8[i]*8, [4x4], [12x4], ...,[252x4], [4x12], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[23.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride8[i]*8, [4x4], [12x4], ...,[188x4], [4x12], ...
	AnchorMatrixConstruct(m_stride8_anchros, YOLOV8N_OD_STRIDE_8, m_stride8_total_anchors);
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[15.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride16[i]*16, [8x8], [24x8], ...,[248x8], [8x24], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[11.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride16[i]*16, [8x8], [24x8], ...,[184x8], [8x24], ...
	AnchorMatrixConstruct(m_stride16_anchros, YOLOV8N_OD_STRIDE_16, m_stride16_total_anchors);
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[7.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride32[i]*32, [16x16], [48x16], ...,[240x16], [16x48], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[5.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride32[i]*32, [16x16], [48x16], ...,[176x16], [16x48], ...
	AnchorMatrixConstruct(m_stride32_anchros, YOLOV8N_OD_STRIDE_32, m_stride32_total_anchors);

    if (!ResolveOutputTensorMapping())
    {
        printf("YOLOv8nODPostProcessing(): failed to resolve output tensor mapping\n");
    }
}

void YOLOv8nODPostProcessing::RunPostProcessing(
    uint32_t imgNetCols,
    uint32_t imgNetRows,
    uint32_t imgSrcCols,
    uint32_t imgSrcRows,
    std::vector<DetectionResult> &resultsOut    /* init postprocessing */
)
{
    float fXScale = (float)imgSrcCols / (float)imgNetCols; 
    float fYScale = (float)imgSrcRows / (float)imgNetRows;
	int i;
	
	std::forward_list<Detection> sDetections;
	GetNetworkBoxes(sDetections);
	CalculateNMS(sDetections, YOLOV8N_OD_CLASS, 0.45);
	
	resultsOut.clear();

	float score = 0.0;

	for (auto box=sDetections.begin(); box != sDetections.end(); ++box) {

		score = box->prob[box->cls]; 

		if(score > 0)
		{
			const S_DETECTION_BOX detectBox = ClampAndValidateDetectionBox(
				box->bbox.x * fXScale,
				box->bbox.y * fYScale,
				box->bbox.w * fXScale,
				box->bbox.h * fYScale,
				box->cls,
				score,
				imgSrcCols,
				imgSrcRows);

			if (IsValidDetectionBox(detectBox))
			{
				DetectionResult detectResult(detectBox);
				resultsOut.push_back(detectResult);
			}
		}
	}
}

void YOLOv8nODPostProcessing::GetNetworkBoxes(
        std::forward_list<Detection>& detections)
{
    if (!m_outputTensorMapping.valid && !ResolveOutputTensorMapping())
    {
        printf("GetNetworkBoxes(): output tensor mapping unavailable\n");
        return;
    }

	TfLiteTensor* psConfidenceTensor;
	TfLiteTensor* psBoxTensor;
	
	psConfidenceTensor = m_model->GetOutputTensor(m_outputTensorMapping.stride8Confidence);
	psBoxTensor = m_model->GetOutputTensor(m_outputTensorMapping.stride8Box);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride8_anchros, YOLOV8N_OD_STRIDE_8, m_stride8_total_anchors, m_threshold, detections); 

	psConfidenceTensor = m_model->GetOutputTensor(m_outputTensorMapping.stride16Confidence);
	psBoxTensor = m_model->GetOutputTensor(m_outputTensorMapping.stride16Box);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride16_anchros, YOLOV8N_OD_STRIDE_16, m_stride16_total_anchors, m_threshold, detections); 

	psConfidenceTensor = m_model->GetOutputTensor(m_outputTensorMapping.stride32Confidence);
	psBoxTensor = m_model->GetOutputTensor(m_outputTensorMapping.stride32Box);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride32_anchros, YOLOV8N_OD_STRIDE_32, m_stride32_total_anchors, m_threshold, detections); 

}

bool YOLOv8nODPostProcessing::ResolveOutputTensorMapping()
{
    std::vector<OutputTensorShape> shapes;
    shapes.reserve(m_model->GetNumOutputs());

    for (size_t i = 0; i < m_model->GetNumOutputs(); ++i)
    {
        TfLiteTensor* tensor = m_model->GetOutputTensor(i);
        if (!tensor || !tensor->dims || tensor->dims->size < 3)
        {
            continue;
        }

        shapes.push_back(
            {
                static_cast<int>(i),
                tensor->dims->size,
                tensor->dims->data[1],
                tensor->dims->data[2],
            });
    }

    m_outputTensorMapping = arm::app::yolov8n_od::ResolveOutputTensorMapping(
        shapes,
        YOLOV8N_OD_CLASS,
        m_stride8_total_anchors,
        m_stride16_total_anchors,
        m_stride32_total_anchors);

    if (!m_outputTensorMapping.valid)
    {
        printf("ResolveOutputTensorMapping(): unable to match YOLO raw outputs\n");
        return false;
    }

    return true;
}

} /* namespace YOLOv8nODPostProcessing */
} /* namespace app */
} /* namespace arm */
	
