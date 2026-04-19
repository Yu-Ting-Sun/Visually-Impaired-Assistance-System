/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Object tracker network sample. Demonstrate multi-object tracking.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#define FDOWNSAMPLE_W 80
#define FDOWNSAMPLE_H 60
#define ZONE_COUNT 3

#define KNOWN_ALERT_PRINT_INTERVAL_FRAMES 10

#define UNKNOWN_MOTION_PIXEL_THRESHOLD 12
#define UNKNOWN_ZONE_RATIO_THRESHOLD 0.08f
#define UNKNOWN_ENTER_FRAMES 3
#define UNKNOWN_EXIT_FRAMES 4
#define UNKNOWN_REPRINT_INTERVAL_FRAMES 30

__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t prev_frame[FDOWNSAMPLE_W * FDOWNSAMPLE_H];
static bool prev_frame_valid = false;
static uint32_t g_frame_seq = 0;
static const char* g_zone_names[ZONE_COUNT] = {"LEFT", "CENTER", "RIGHT"};
static const char* g_danger_level_names[3] = {"SAFE", "CAUTION", "DANGER"};
static uint8_t g_unknown_active[ZONE_COUNT] = {0, 0, 0};
static uint8_t g_unknown_enter_streak[ZONE_COUNT] = {0, 0, 0};
static uint8_t g_unknown_exit_streak[ZONE_COUNT] = {0, 0, 0};
static uint32_t g_unknown_last_print_frame[ZONE_COUNT] = {0, 0, 0};
#include "BoardInit.hpp"      /* Board initialisation */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "YOLOv8nODModel.hpp"       /* Model API */
#include "YOLOv8nODPostProcessing.hpp"
#include "Labels.hpp"
#include "WarningLogic.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"
#include "ModelFileReader.h"
#include "ff.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_CCAP__
#define __USE_DISPLAY__
//#define __USE_UVC__

#include "Profiler.hpp"

#if defined (__USE_CCAP__)
#include "ImageSensor.h"
#endif

#if !defined (__USE_CCAP__)
#include "InputFiles.hpp"             /* Baked-in input (not needed for live data) */
#endif

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#define IMAGE_REAL_FRAMRATE		16  
#include "BYTETracker.h"

#include "log_macros.h"      /* Logging macros (optional) */

#define NUM_FRAMEBUF 2  //1 or 2

#define MODEL_AT_HYPERRAM_ADDR (0x82400000)

#define OD_PRESENCE_THRESHOLD  				(0.5)
#define ENABLE_UNKNOWN_EXPERIMENT           (0)

typedef enum
{
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct
{
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
    std::vector<arm::app::yolov8n_od::DetectionResult> results;
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

} /* namespace app */
} /* namespace arm */

//frame buffer managemnet function
static S_FRAMEBUF *get_empty_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF)
            return &s_asFramebuf[i];
    }

    return NULL;
}

#define IMAGE_DISP_UPSCALE_FACTOR 1
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

/* Image processing initiate function */
//Used by omv library
#if defined(__USE_UVC__)
//UVC only support QVGA, QQVGA
#define GLCD_WIDTH	320
#define GLCD_HEIGHT	240
#elif !defined(__USE_CCAP__)
#define GLCD_WIDTH		IMAGE_WIDTH
#define GLCD_HEIGHT		IMAGE_HEIGHT
#else
#define GLCD_WIDTH		320
#define GLCD_HEIGHT		240
#endif

//RGB565
#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".bss.vram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];
#endif

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0 ; i < NUM_FRAMEBUF; i++)
    {
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;
    }

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

static inline uint8_t RGB565ToGray(uint16_t pixel)
{
    const int r5 = (pixel >> 11) & 0x1F;
    const int g6 = (pixel >> 5) & 0x3F;
    const int b5 = pixel & 0x1F;

    const int r8 = (r5 << 3) | (r5 >> 2);
    const int g8 = (g6 << 2) | (g6 >> 4);
    const int b8 = (b5 << 3) | (b5 >> 2);

    return (uint8_t)((77 * r8 + 150 * g8 + 29 * b8) >> 8);
}

static constexpr arm::app::warning::DebugViewMode kDebugViewMode =
    arm::app::warning::DebugViewMode::RawWithTracks;
static arm::app::warning::WarningEvent g_last_warning_event;

static const char *GetSafeLabel(const std::vector<std::string> &labels, int classId)
{
    if (classId >= 0 && classId < static_cast<int>(labels.size()))
    {
        return labels[classId].c_str();
    }

    return "unknown";
}

static std::vector<Object> ConvertDetectionsToTrackObjects(
    const std::vector<arm::app::yolov8n_od::DetectionResult> &results,
    std::vector<arm::app::warning::FrameObject> *rawWarningObjects = nullptr
)
{
    std::vector<struct Object> detObjects;
    detObjects.reserve(results.size());

    if (rawWarningObjects)
    {
        rawWarningObjects->clear();
        rawWarningObjects->reserve(results.size());
    }

    for (size_t p = 0; p < results.size(); ++p)
    {
        const arm::app::yolov8n_od::DetectionResult &detectBox = results[p];
        struct Object detObject;

        detObject.rect.x = detectBox.m_detectBox.x;
        detObject.rect.y = detectBox.m_detectBox.y;
        detObject.rect.w = detectBox.m_detectBox.w;
        detObject.rect.h = detectBox.m_detectBox.h;
        detObject.label = detectBox.m_detectBox.cls;
        detObject.prob = detectBox.m_detectBox.normalisedVal;
        detObjects.push_back(detObject);

        if (rawWarningObjects)
        {
            arm::app::warning::FrameObject rawObject;
            rawObject.id = static_cast<int>(p);
            rawObject.source = arm::app::warning::WarningSource::RawFallback;
            rawObject.class_id = detectBox.m_detectBox.cls;
            rawObject.bbox = detectBox.m_detectBox;
            rawObject.score = detectBox.m_detectBox.normalisedVal;
            rawWarningObjects->push_back(rawObject);
        }
    }

    return detObjects;
}

static std::vector<arm::app::warning::FrameObject> ConvertTracksToWarningObjects(
    const std::vector<STrack> &tracks)
{
    std::vector<arm::app::warning::FrameObject> warningObjects;
    warningObjects.reserve(tracks.size());

    for (const auto &track : tracks)
    {
        arm::app::warning::FrameObject object;
        object.id = track.track_id;
        object.source = arm::app::warning::WarningSource::Tracker;
        object.class_id = track.class_id;
        object.bbox.x = static_cast<int>(track.tlwh[0]);
        object.bbox.y = static_cast<int>(track.tlwh[1]);
        object.bbox.w = static_cast<int>(track.tlwh[2]);
        object.bbox.h = static_cast<int>(track.tlwh[3]);
        object.bbox.cls = track.class_id;
        object.bbox.normalisedVal = track.score;
        object.score = track.score;
        warningObjects.push_back(object);
    }

    return warningObjects;
}

static void DrawRawDetections(
    const std::vector<arm::app::yolov8n_od::DetectionResult> &results,
    image_t *drawImg)
{
    const int rawColor = COLOR_R8_G8_B8_TO_RGB565(0, 160, 255);
    for (const auto &detectBox : results)
    {
        imlib_draw_rectangle(drawImg,
                             detectBox.m_detectBox.x,
                             detectBox.m_detectBox.y,
                             detectBox.m_detectBox.w,
                             detectBox.m_detectBox.h,
                             rawColor,
                             1,
                             false);
    }
}

static void DrawTrackedDetections(
    const std::vector<STrack> &tracks,
    image_t *drawImg,
    const std::vector<std::string> &labels)
{
    char szDisplayText[100];

    for (const auto &track : tracks)
    {
        const vector<float> &tlwh = track.tlwh;
        const int colorIdx = track.track_id + 3;
        const int trackColor = COLOR_R8_G8_B8_TO_RGB565(37 * colorIdx % 255, 17 * colorIdx % 255, 29 * colorIdx % 255);

        sprintf(szDisplayText, "%d: %s", track.track_id, GetSafeLabel(labels, track.class_id));
        imlib_draw_rectangle(drawImg, (int)tlwh[0], (int)tlwh[1], (int)tlwh[2], (int)tlwh[3], trackColor, 2, false);
        imlib_draw_string(drawImg, (int)tlwh[0], (int)tlwh[1] - 16, szDisplayText, trackColor, 2, 0, 0, false,
                          false, false, false, 0, false, false);
    }
}

static void DrawWarningHighlight(
    const arm::app::warning::WarningEvent &event,
    image_t *drawImg,
    const std::vector<std::string> &labels)
{
    if (!event.has_candidate)
    {
        return;
    }

    const int color =
        (event.severity == arm::app::warning::WarningSeverity::Danger)
            ? COLOR_R8_G8_B8_TO_RGB565(255, 64, 64)
            : COLOR_R8_G8_B8_TO_RGB565(255, 220, 0);

    char szDisplayText[120];
    sprintf(szDisplayText,
            "%s %s",
            arm::app::warning::ToString(event.severity),
            GetSafeLabel(labels, event.class_id));

    imlib_draw_rectangle(drawImg,
                         event.bbox.x,
                         event.bbox.y,
                         event.bbox.w,
                         event.bbox.h,
                         color,
                         3,
                         false);
    imlib_draw_string(drawImg,
                      event.bbox.x,
                      std::max(0, event.bbox.y - 18),
                      szDisplayText,
                      color,
                      2,
                      0,
                      0,
                      false,
                      false,
                      false,
                      false,
                      0,
                      false,
                      false);
}

static void LogWarningEvent(
    const arm::app::warning::WarningEvent &event,
    const std::vector<std::string> &labels)
{
    if (!event.emitted)
    {
        return;
    }

    printf("[WARN][%s][%s] %s %s risk=%.2f overlap=%.2f\n",
           arm::app::warning::ToString(event.source),
           g_zone_names[event.zone],
           arm::app::warning::ToString(event.severity),
           GetSafeLabel(labels, event.class_id),
           event.risk_score,
           event.path_overlap);
}

static arm::app::warning::WarningEvent RenderDetectionsAndWarnings(
    const std::vector<arm::app::yolov8n_od::DetectionResult> &results,
    image_t *drawImg,
    const std::vector<std::string> &labels,
    BYTETracker *tracker,
    arm::app::warning::WarningEngine *warningEngine)
{
    std::vector<arm::app::warning::FrameObject> rawWarningObjects;
    std::vector<Object> detObjects = ConvertDetectionsToTrackObjects(results, &rawWarningObjects);
    std::vector<STrack> output_stracks = tracker->update(detObjects);
    std::vector<arm::app::warning::FrameObject> trackerWarningObjects = ConvertTracksToWarningObjects(output_stracks);

    if (kDebugViewMode == arm::app::warning::DebugViewMode::RawOnly ||
        kDebugViewMode == arm::app::warning::DebugViewMode::RawWithTracks)
    {
        DrawRawDetections(results, drawImg);
    }

    if (kDebugViewMode == arm::app::warning::DebugViewMode::RawWithTracks)
    {
        DrawTrackedDetections(output_stracks, drawImg, labels);
    }

    arm::app::warning::WarningEvent event = warningEngine->Update(g_frame_seq, trackerWarningObjects, rawWarningObjects);
    DrawWarningHighlight(event, drawImg, labels);
    LogWarningEvent(event, labels);
    return event;
}

static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\YOLOv8n-od.tflite"
#define EACH_READ_SIZE 512
	
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */	
    f_chdrive(sd_path);          /* set default path */

	int32_t i32FileSize;
	int32_t i32FileReadIndex = 0;
	int32_t i32Read;
	
	if(!ModelFileReader_Initialize(MODEL_FILE))
	{
        printf_err("Unable open model %s\n", MODEL_FILE);		
		return -1;
	}
	
	i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		i32FileReadIndex += i32Read;
	}
	
	if(i32FileReadIndex < i32FileSize)
	{
        printf_err("Read Model file size is not enough\n");		
		return -2;
	}
	
#if 0
	/* verify */
	i32FileReadIndex = 0;
	ModelFileReader_Rewind();
	BYTE au8TempBuf[EACH_READ_SIZE];
	
	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)au8TempBuf, EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		
		if(std::memcmp(au8TempBuf, (void *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), i32Read)!= 0)
		{
			printf_err("verify the model file content is incorrect at %i \n", i32FileReadIndex);		
			return -3;
		}
		i32FileReadIndex += i32Read;
	}
	
#endif	
	ModelFileReader_Finish();
	
	return i32FileSize;
}	

int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

	/* Copy model file from SD to HyperRAM*/
	int32_t i32ModelSize;
		
	i32ModelSize = PrepareModelToHyperRAM();

	if(i32ModelSize <= 0 )
	{
        printf_err("Failed to prepare model\n");
        return 1;
	}

    /* Model object creation and initialisation. */
    arm::app::YOLOv8nODModel model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena) + ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#if (NUM_FRAMEBUF == 2)
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)frame_buf1),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#endif
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

#if !defined (__USE_CCAP__)
    uint8_t u8ImgIdx = 0;
    char chStdIn;
#endif

    TfLiteTensor *inputTensor   = model.GetInputTensor(0);

    if (!inputTensor->dims)
    {
        printf_err("Invalid input tensor dims\n");
        return 2;
    }
    else if (inputTensor->dims->size < 3)
    {
        printf_err("Input tensor dimension should be >= 3\n");
        return 3;
    }

    TfLiteIntArray *inputShape = model.GetInputShape(0);

    const int inputImgCols = inputShape->data[arm::app::YOLOv8nODModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::YOLOv8nODModel::ms_inputRowsIdx];
    const uint32_t nChannels = inputShape->data[arm::app::YOLOv8nODModel::ms_inputChannelsIdx];

    /* Get input quantization params information. */
    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);
	
    // postProcess
    arm::app::yolov8n_od::YOLOv8nODPostProcessing postProcess(&model, OD_PRESENCE_THRESHOLD);

    //label information
    std::vector<std::string> labels;
    GetLabelsVector(labels);
	
    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

#if defined(__PROFILE__)

    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

#if defined (__USE_CCAP__)
    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);
#endif

#if defined (__USE_DISPLAY__)
    char szDisplayText[100];
    char szSidePanelText[64];
    char szDebugText[64];
    S_DISP_RECT sDispRect;
    S_DISP_RECT sSidePanelRect;
    uint32_t u32LcdWidth;
    uint32_t u32LcdHeight;
    uint32_t u32RenderedImgWidth;
    uint32_t u32RenderedImgHeight;
    uint32_t u32SidePanelWidth = 0;
    uint32_t u32SidePanelHeight = 0;
    int i32SideTextScale = 1;
    uint32_t u32SideLineStep = FONT_HTIGHT;
    bool bEnableSidePanel = false;

    Display_Init();
    Display_ClearLCD(C_WHITE);

    u32LcdWidth = Disaplay_GetLCDWidth();
    u32LcdHeight = Disaplay_GetLCDHeight();
    u32RenderedImgWidth = frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR;
    u32RenderedImgHeight = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;

    if (u32LcdWidth > (u32RenderedImgWidth + (FONT_WIDTH * 8)))
    {
        const int desiredScale = 8;
        const int longestTextChars = 26; // "W:DANGER CENTER microwave"
        const int panelLines = 3;        // STATUS, warning, debug
        int maxScaleByWidth;
        int maxScaleByHeight;
        int fitScale;

        bEnableSidePanel = true;
        u32SidePanelWidth = u32LcdWidth - u32RenderedImgWidth;
        u32SidePanelHeight = u32LcdHeight;

        maxScaleByWidth = (int)((u32SidePanelWidth > 8 ? (u32SidePanelWidth - 8) : u32SidePanelWidth) / (FONT_WIDTH * longestTextChars));
        maxScaleByHeight = (int)((u32SidePanelHeight > 8 ? (u32SidePanelHeight - 8) : u32SidePanelHeight) / (FONT_HTIGHT * panelLines));
        fitScale = maxScaleByWidth;
        if (maxScaleByHeight < fitScale)
            fitScale = maxScaleByHeight;
        if (fitScale < 1)
            fitScale = 1;
        if (fitScale > desiredScale)
            fitScale = desiredScale;

        i32SideTextScale = fitScale;
        u32SideLineStep = FONT_HTIGHT * i32SideTextScale;

        sSidePanelRect.u32TopLeftX = u32RenderedImgWidth;
        sSidePanelRect.u32TopLeftY = 0;
        sSidePanelRect.u32BottonRightX = u32LcdWidth - 1;
        sSidePanelRect.u32BottonRightY = u32LcdHeight - 1;
        Display_ClearRect(C_WHITE, &sSidePanelRect);
    }
#endif

#if defined (__USE_UVC__)
	UVC_Init();
    HSUSBD_Start();
#endif

    BYTETracker tracker(IMAGE_REAL_FRAMRATE, 30);
    arm::app::warning::WarningEngine warningEngine(frameBuffer.w, frameBuffer.h);
	
	while(1)
	{
        emptyFramebuf = get_empty_framebuf();

        if (emptyFramebuf)
        {
#if defined (__USE_CCAP__)
            //capture frame from CCAP
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif

            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
#endif
		}

        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
            //resize full image to input tensor
            image_t resizeImg;

            roi.x = 0;
            roi.y = 0;
            roi.w = fullFramebuf->frameImage.w;
            roi.h = fullFramebuf->frameImage.h;

            resizeImg.w = inputImgCols;
            resizeImg.h = inputImgRows;
            resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
            resizeImg.pixfmt = PIXFORMAT_RGB888;

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
            imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg, &roi);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("resize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
			//Quantize input tensor data
			auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
			auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

			for (size_t i = 0; i < inputTensor->bytes; i++)
			{
//				auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
//				signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
				signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
			}

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
			profiler.StartProfiling("Inference");
#endif

			model.RunInference();

#if defined(__PROFILE__)
			profiler.StopProfiling();
			profiler.PrintProfilingResult();
#endif

            fullFramebuf->eState = eFRAMEBUF_INF;
        }
		
        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {
			//post process

#if defined(__PROFILE__)
			u64StartCycle = pmu_get_systick_Count();
#endif
			postProcess.RunPostProcessing(
				inputImgCols,
				inputImgRows,
				infFramebuf->frameImage.w,
				infFramebuf->frameImage.h,
				infFramebuf->results);

            g_frame_seq++;
            g_last_warning_event = arm::app::warning::WarningEvent{};

#if defined(__PROFILE__)
			u64EndCycle = pmu_get_systick_Count();
			info("post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

            //draw bbox and render
            {
#if defined(__PROFILE__)
				u64StartCycle = pmu_get_systick_Count();
#endif

				g_last_warning_event = RenderDetectionsAndWarnings(
                    infFramebuf->results,
                    &infFramebuf->frameImage,
                    labels,
                    &tracker,
                    &warningEngine);

#if defined(__PROFILE__)
				u64EndCycle = pmu_get_systick_Count();
				info("draw box cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
			}

            //display result image
#if defined (__USE_DISPLAY__)
            //Display image on LCD
            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
			sDispRect.u32BottonRightX = ((frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
			sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
#if ENABLE_UNKNOWN_EXPERIMENT
            // ===== Frame Difference 幀差法（低解析度版）=====
            if (prev_frame_valid) {
                uint16_t *curr = (uint16_t *)infFramebuf->frameImage.data;
                int diff_count[ZONE_COUNT] = {0, 0, 0};
                const int zone_w = FDOWNSAMPLE_W / ZONE_COUNT;

                // 將「已知物件」先投影到低解析度遮罩，避免 known 物件觸發 unknown 警示。
                uint8_t known_mask[FDOWNSAMPLE_W * FDOWNSAMPLE_H];
                std::memset(known_mask, 0, sizeof(known_mask));

                for (size_t r = 0; r < infFramebuf->results.size(); r++) {
                    const arm::app::yolov8n_od::DetectionResult &det = infFramebuf->results[r];

                    int bx0 = (int)det.m_detectBox.x;
                    int by0 = (int)det.m_detectBox.y;
                    int bx1 = bx0 + (int)det.m_detectBox.w - 1;
                    int by1 = by0 + (int)det.m_detectBox.h - 1;

                    if (bx1 < 0 || by1 < 0 || bx0 >= GLCD_WIDTH || by0 >= GLCD_HEIGHT)
                        continue;

                    if (bx0 < 0) bx0 = 0;
                    if (by0 < 0) by0 = 0;
                    if (bx1 >= GLCD_WIDTH) bx1 = GLCD_WIDTH - 1;
                    if (by1 >= GLCD_HEIGHT) by1 = GLCD_HEIGHT - 1;

                    int lx0 = (bx0 * FDOWNSAMPLE_W) / GLCD_WIDTH;
                    int ly0 = (by0 * FDOWNSAMPLE_H) / GLCD_HEIGHT;
                    int lx1 = (bx1 * FDOWNSAMPLE_W) / GLCD_WIDTH;
                    int ly1 = (by1 * FDOWNSAMPLE_H) / GLCD_HEIGHT;

                    if (lx0 < 0) lx0 = 0;
                    if (ly0 < 0) ly0 = 0;
                    if (lx1 >= FDOWNSAMPLE_W) lx1 = FDOWNSAMPLE_W - 1;
                    if (ly1 >= FDOWNSAMPLE_H) ly1 = FDOWNSAMPLE_H - 1;

                    for (int y = ly0; y <= ly1; y++) {
                        for (int x = lx0; x <= lx1; x++) {
                            known_mask[y * FDOWNSAMPLE_W + x] = 1;
                        }
                    }
                }

                for (int y = 0; y < FDOWNSAMPLE_H; y++) {
                    for (int x = 0; x < FDOWNSAMPLE_W; x++) {
                        const int mask_idx = y * FDOWNSAMPLE_W + x;

                        if (known_mask[mask_idx]) {
                            continue;
                        }

                        const int src_x = (x * GLCD_WIDTH) / FDOWNSAMPLE_W;
                        const int src_y = (y * GLCD_HEIGHT) / FDOWNSAMPLE_H;

                        const uint16_t curr_px = curr[src_y * GLCD_WIDTH + src_x];
                        const uint8_t curr_gray = RGB565ToGray(curr_px);
                        const uint8_t prev_gray = prev_frame[mask_idx];
                        const int diff = abs((int)curr_gray - (int)prev_gray);

                        if (diff > UNKNOWN_MOTION_PIXEL_THRESHOLD) {
                            int zone = x / zone_w;
                            if (zone >= ZONE_COUNT) zone = ZONE_COUNT - 1;
                            diff_count[zone]++;
                        }
                    }
                }

                const int threshold = (int)((FDOWNSAMPLE_W * FDOWNSAMPLE_H / ZONE_COUNT) * UNKNOWN_ZONE_RATIO_THRESHOLD);

                for (int z = 0; z < ZONE_COUNT; z++) {
                    const bool is_over_threshold = (diff_count[z] > threshold);

                    if (is_over_threshold) {
                        if (g_unknown_enter_streak[z] < 255) g_unknown_enter_streak[z]++;
                        g_unknown_exit_streak[z] = 0;
                    } else {
                        g_unknown_enter_streak[z] = 0;
                        if (g_unknown_exit_streak[z] < 255) g_unknown_exit_streak[z]++;
                    }

                    if (!g_unknown_active[z] && g_unknown_enter_streak[z] >= UNKNOWN_ENTER_FRAMES) {
                        g_unknown_active[z] = 1;
                        printf("[UNKNOWN OBSTACLE] %s\n", g_zone_names[z]);
                        g_unknown_last_print_frame[z] = g_frame_seq;
                    } else if (g_unknown_active[z] && g_unknown_exit_streak[z] >= UNKNOWN_EXIT_FRAMES) {
                        g_unknown_active[z] = 0;
                    } else if (g_unknown_active[z] && ((g_frame_seq - g_unknown_last_print_frame[z]) >= UNKNOWN_REPRINT_INTERVAL_FRAMES)) {
                        printf("[UNKNOWN OBSTACLE] %s\n", g_zone_names[z]);
                        g_unknown_last_print_frame[z] = g_frame_seq;
                    }
                }
            }

            // 儲存縮圖
            uint16_t *curr = (uint16_t *)infFramebuf->frameImage.data;
            for (int y = 0; y < FDOWNSAMPLE_H; y++) {
                for (int x = 0; x < FDOWNSAMPLE_W; x++) {
                    const int src_x = (x * GLCD_WIDTH) / FDOWNSAMPLE_W;
                    const int src_y = (y * GLCD_HEIGHT) / FDOWNSAMPLE_H;
                    prev_frame[y * FDOWNSAMPLE_W + x] = RGB565ToGray(curr[src_y * GLCD_WIDTH + src_x]);
                }
            }
            prev_frame_valid = true;
            // ===== END Frame Difference =====         
#endif

            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect, IMAGE_DISP_UPSCALE_FACTOR);

            if (bEnableSidePanel)
            {
                uint32_t lineY = 4;
                uint32_t textColor = C_BLACK;

                Display_ClearRect(C_WHITE, &sSidePanelRect);

                Display_PutText("STATUS", 6, sSidePanelRect.u32TopLeftX + 4, lineY, C_BLUE, C_WHITE, false, i32SideTextScale);
                lineY += u32SideLineStep;

                if (g_last_warning_event.has_candidate)
                {
                    sprintf(szSidePanelText,
                            "W:%s %s",
                            arm::app::warning::ToString(g_last_warning_event.severity),
                            g_zone_names[g_last_warning_event.zone]);

                    if (g_last_warning_event.severity == arm::app::warning::WarningSeverity::Danger)
                        textColor = C_RED;
                    else if (g_last_warning_event.severity == arm::app::warning::WarningSeverity::Caution)
                        textColor = C_YELLOW;
                    else
                        textColor = C_GREEN;
                }
                else
                {
                    sprintf(szSidePanelText, "W:NONE");
                    textColor = C_BLACK;
                }

                Display_PutText(szSidePanelText, strlen(szSidePanelText), sSidePanelRect.u32TopLeftX + 4, lineY, textColor, C_WHITE, false, i32SideTextScale);
                lineY += u32SideLineStep;

                std::memset(szDebugText, 0, sizeof(szDebugText));
                if (g_last_warning_event.has_candidate)
                {
                    sprintf(szDebugText,
                            "SRC:%s %s",
                            arm::app::warning::ToString(g_last_warning_event.source),
                            GetSafeLabel(labels, g_last_warning_event.class_id));
                    textColor = C_BLUE;
                }
                else
                {
                    sprintf(szDebugText, "DBG:RAW+TRK");
                    textColor = C_BLACK;
                }

                Display_PutText(szDebugText, strlen(szDebugText), sSidePanelRect.u32TopLeftX + 4, lineY, textColor, C_WHITE, false, i32SideTextScale);
            }

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#endif

#if defined (__USE_UVC__)
			if(UVC_IsConnect())
			{
#if (UVC_Color_Format == UVC_Format_YUY2)
				image_t RGB565Img;
				image_t YUV422Img;

				RGB565Img.w = infFramebuf->frameImage.w;
				RGB565Img.h = infFramebuf->frameImage.h;
				RGB565Img.data = (uint8_t *)infFramebuf->frameImage.data;
				RGB565Img.pixfmt = PIXFORMAT_RGB565;

				YUV422Img.w = RGB565Img.w;
				YUV422Img.h = RGB565Img.h;
				YUV422Img.data = (uint8_t *)infFramebuf->frameImage.data;
				YUV422Img.pixfmt = PIXFORMAT_YUV422;
				
				roi.x = 0;
				roi.y = 0;
				roi.w = RGB565Img.w;
				roi.h = RGB565Img.h;
				imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);
				
#else
				image_t origImg;
				image_t vflipImg;

				origImg.w = infFramebuf->frameImage.w;
				origImg.h = infFramebuf->frameImage.h;
				origImg.data = (uint8_t *)infFramebuf->frameImage.data;
				origImg.pixfmt = PIXFORMAT_RGB565;

				vflipImg.w = origImg.w;
				vflipImg.h = origImg.h;
				vflipImg.data = (uint8_t *)infFramebuf->frameImage.data;
				vflipImg.pixfmt = PIXFORMAT_RGB565;

				imlib_nvt_vflip(&origImg, &vflipImg);
#endif
				UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE, uvcStatus.StillImage);				

			}

#endif

            u64PerfFrames ++;
			if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);
                //sprintf(szDisplayText,"Time %llu",(uint64_t) pmu_get_systick_Count() / (uint64_t)SystemCoreClock);
                //info("Running %s sec \n", szDisplayText);

                sDispRect.u32TopLeftX = 0;
				sDispRect.u32TopLeftY = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
				sDispRect.u32BottonRightX = (frameBuffer.w);
				sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
                    frameBuffer.h,
                    C_BLUE,
                    C_WHITE,
                    false,
					FONT_DISP_UPSCALE_FACTOR
                );
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
			}

            infFramebuf->eState = eFRAMEBUF_EMPTY;
		}
		
		//Wait CCAP ready
		if (emptyFramebuf)
		{
#if !defined (__USE_CCAP__)
            info("Press 'n' to run next image inference \n");
            info("Press 'q' to exit program \n");

            while ((chStdIn = getchar()))
            {
                if (chStdIn == 'q')
                {
                    return 0;
                }
                else if (chStdIn != 'n')
                {
                    break;
                }
            }

            const uint8_t *pu8ImgSrc = get_img_array(u8ImgIdx);

            if (nullptr == pu8ImgSrc)
            {
                printf_err("Failed to get image index %" PRIu32 " (max: %u)\n", u8ImgIdx,
                           NUMBER_OF_FILES - 1);
                return -1;
            }

            u8ImgIdx ++;

            if (u8ImgIdx >= NUMBER_OF_FILES)
                u8ImgIdx = 0;

#endif

#if defined (__USE_CCAP__)
			//Capture new image

			ImageSensor_WaitCaptureDone();
#if defined(__PROFILE__)
			u64CCAPEndCycle = pmu_get_systick_Count();
			info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif
#else
            //copy source image to frame buffer
            image_t srcImg;

            srcImg.w = IMAGE_WIDTH;
            srcImg.h = IMAGE_HEIGHT;
            srcImg.data = (uint8_t *)pu8ImgSrc;
            srcImg.pixfmt = PIXFORMAT_RGB888;

            roi.x = 0;
            roi.y = 0;
            roi.w = IMAGE_WIDTH;
            roi.h = IMAGE_HEIGHT;

            imlib_nvt_scale(&srcImg, &emptyFramebuf->frameImage, &roi);

#endif
			emptyFramebuf->eState = eFRAMEBUF_FULL;		
		}

	}
	
    return 0;
}
