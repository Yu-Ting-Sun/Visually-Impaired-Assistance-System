/**************************************************************************//**
 * @file     VoicePlayer.cpp
 * @brief    I2S + NAU8822 voice clip player with segment queue.
 *
 * Design:
 *   - Voice files: 16 kHz / 16-bit / mono WAV on SD card under 0:/voice/.
 *   - Output:      I2S0 -> NAU8822 -> 3.5mm jack on NuMaker-M55M1.
 *   - Transport:   PDMA0 ch 7, scatter-gather ping-pong between two 16-bit
 *                  stereo PCM buffers; worker pump polls TX-done flags and
 *                  refills the next free buffer from the current WAV file.
 *   - No ISR:      polling avoids conflicting with drv_pdma.c's shared
 *                  PDMA0_IRQHandler. We also reserve our channel in the
 *                  nu_pdma framework's free mask so it won't re-allocate it.
 ******************************************************************************/
#include "VoicePlayer.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "NuMicro.h"
#include "ff.h"

/* ARMCLANG promotes -Wregister to error; the M55M1 __PC() macro (pulled in
 * by PD/PG/PI port macros) uses the deprecated 'register' keyword. Silence it
 * only for this file. */
#pragma clang diagnostic ignored "-Wregister"

/* ---------------- configuration ---------------- */

#define VP_I2C_PORT        I2C3
#define VP_PDMA            PDMA0
#define VP_PDMA_CH         7                       /* reserve from nu_pdma */
#define VP_PDMA_CH_MASK    (1u << VP_PDMA_CH)

#define VP_SAMPLE_RATE     16000
#define VP_PCM_WORDS       2048                    /* 32-bit words per buffer (L+R packed) -> 2048*4 = 8 KB per buffer */
#define VP_QUEUE_MAX       12
#define VP_PATH_MAX        40

/* NOTE on PDMA channel choice:
 *   drv_pdma.c's nu_pdma framework allocates channels dynamically from low to
 *   high (channels 0, 1, 2 ... first). Display + memcpy typically consume
 *   only a handful, so ch 7 is effectively reserved for us. We configure it
 *   via the raw PDMA_* API below; the nu_pdma framework's IRQ handler
 *   dispatches per-channel callbacks, and since we never register one for
 *   ch 7 it simply ignores our TD events — we poll them ourselves. */

/* ---------------- DMA descriptors / PCM buffers ---------------- */

typedef struct {
    uint32_t ctl;
    uint32_t endsrc;
    uint32_t enddest;
    uint32_t offset;
} vp_dma_desc_t;

#if (NVT_DCACHE_ON == 1)
NVT_DTCM __ALIGNED(32) static vp_dma_desc_t s_desc[2];
__attribute__((aligned(32))) static int32_t s_pcm[2][VP_PCM_WORDS];
#else
static vp_dma_desc_t s_desc[2] __attribute__((aligned(32)));
static int32_t s_pcm[2][VP_PCM_WORDS] __attribute__((aligned(32)));
#endif

/* ---------------- queue / state ---------------- */

static char     s_queue[VP_QUEUE_MAX][VP_PATH_MAX];
static uint8_t  s_q_head = 0;           /* next item to play */
static uint8_t  s_q_tail = 0;           /* next free slot    */
static bool     s_initialised = false;
static bool     s_playing = false;      /* I2S TX DMA currently running */
static bool     s_file_open = false;
static FIL      s_fil;
static uint8_t  s_buf_full[2] = { 0, 0 };     /* 1 = holds fresh PCM waiting to play */
static uint8_t  s_playing_idx = 0;            /* buffer DMA is currently playing */

static inline uint8_t q_count(void)
{
    return (uint8_t)((s_q_tail + VP_QUEUE_MAX - s_q_head) % VP_QUEUE_MAX);
}

static inline bool q_empty(void) { return s_q_head == s_q_tail; }

/* ---------------- NAU8822 codec (via I2C3) ---------------- */

static void I2C_WriteNAU8822(uint8_t addr, uint16_t data)
{
    I2C_START(VP_I2C_PORT); I2C_WAIT_READY(VP_I2C_PORT);
    I2C_SET_DATA(VP_I2C_PORT, 0x1A << 1);
    I2C_SET_CONTROL_REG(VP_I2C_PORT, I2C_CTL_SI); I2C_WAIT_READY(VP_I2C_PORT);
    I2C_SET_DATA(VP_I2C_PORT, (uint8_t)((addr << 1) | (data >> 8)));
    I2C_SET_CONTROL_REG(VP_I2C_PORT, I2C_CTL_SI); I2C_WAIT_READY(VP_I2C_PORT);
    I2C_SET_DATA(VP_I2C_PORT, (uint8_t)(data & 0xFF));
    I2C_SET_CONTROL_REG(VP_I2C_PORT, I2C_CTL_SI); I2C_WAIT_READY(VP_I2C_PORT);
    I2C_STOP(VP_I2C_PORT);
}

static void NAU8822_ConfigSampleRate_16k(void)
{
    /* PLL: 12.288 MHz */
    I2C_WriteNAU8822(36, 0x008);
    I2C_WriteNAU8822(37, 0x00C);
    I2C_WriteNAU8822(38, 0x093);
    I2C_WriteNAU8822(39, 0x0E9);
    /* 16 kHz sample rate */
    I2C_WriteNAU8822(6, 0x1AD);
    I2C_WriteNAU8822(7, 0x006);
}

static void NAU8822_Setup(void)
{
    I2C_WriteNAU8822(0,  0x000); CLK_SysTickDelay(10000);
    I2C_WriteNAU8822(1,  0x02F);
    I2C_WriteNAU8822(2,  0x1BF);
    I2C_WriteNAU8822(3,  0x07F);
    I2C_WriteNAU8822(4,  0x010);  /* 16-bit, I2S, stereo */
    I2C_WriteNAU8822(5,  0x000);
    I2C_WriteNAU8822(6,  0x14D);
    I2C_WriteNAU8822(7,  0x000);
    I2C_WriteNAU8822(10, 0x008);
    I2C_WriteNAU8822(14, 0x108);
    I2C_WriteNAU8822(15, 0x1EF);
    I2C_WriteNAU8822(16, 0x1EF);
    I2C_WriteNAU8822(44, 0x000);
    I2C_WriteNAU8822(47, 0x050);
    I2C_WriteNAU8822(48, 0x050);
    I2C_WriteNAU8822(50, 0x001);
    I2C_WriteNAU8822(51, 0x001);
}

/* ---------------- HW bring-up ---------------- */

static void vp_mfp_init(void)
{
    /* I2S0 pins (same layout as StdDriver/I2S_WAVEPlayer sample) */
    SET_I2S0_BCLK_PI6();
    SET_I2S0_MCLK_PI7();
    SET_I2S0_DI_PI8();
    SET_I2S0_DO_PI9();
    SET_I2S0_LRCK_PI10();
    PI->SMTEN |= GPIO_SMTEN_SMTEN6_Msk;

    /* I2C3 for codec control */
    SET_I2C3_SDA_PG1();
    SET_I2C3_SCL_PG0();
    PG->SMTEN |= GPIO_SMTEN_SMTEN0_Msk;

    /* Headphone jack enable: JK-EN = PD1 low */
    SET_GPIO_PD1();
    GPIO_SetMode(PD, BIT1, GPIO_MODE_OUTPUT);
    PD1 = 0;
}

static void vp_clk_init(void)
{
    CLK_EnableModuleClock(I2S0_MODULE);
    CLK_EnableModuleClock(I2C3_MODULE);
    CLK_EnableModuleClock(PDMA0_MODULE);
    CLK_SetModuleClock(I2S0_MODULE, CLK_I2SSEL_I2S0SEL_HXT, CLK_I2SDIV_I2S0DIV(2));
}

static void vp_pdma_init(void)
{
    const uint32_t ctl = ((VP_PCM_WORDS - 1) << PDMA_DSCT_CTL_TXCNT_Pos)
                       | PDMA_WIDTH_32 | PDMA_SAR_INC | PDMA_DAR_FIX
                       | PDMA_REQ_SINGLE | PDMA_OP_SCATTER;

    s_desc[0].ctl     = ctl;
    s_desc[0].endsrc  = (uint32_t)&s_pcm[0][0];
    s_desc[0].enddest = (uint32_t)&I2S0->TXFIFO;
    s_desc[0].offset  = (uint32_t)&s_desc[1];

    s_desc[1].ctl     = ctl;
    s_desc[1].endsrc  = (uint32_t)&s_pcm[1][0];
    s_desc[1].enddest = (uint32_t)&I2S0->TXFIFO;
    s_desc[1].offset  = (uint32_t)&s_desc[0];

    PDMA_Open(VP_PDMA, VP_PDMA_CH_MASK);
    PDMA_SetTransferMode(VP_PDMA, VP_PDMA_CH, PDMA_I2S0_TX, 1, (uint32_t)&s_desc[0]);
    /* No IRQ enable: we poll PDMA_GET_TD_STS in the pump. */
}

/* ---------------- queue helpers ---------------- */

static bool q_push(const char *path)
{
    uint8_t nxt = (s_q_tail + 1) % VP_QUEUE_MAX;
    if (nxt == s_q_head) return false;      /* full */
    std::strncpy(s_queue[s_q_tail], path, VP_PATH_MAX - 1);
    s_queue[s_q_tail][VP_PATH_MAX - 1] = 0;
    s_q_tail = nxt;
    return true;
}

static bool q_peek(char *out)
{
    if (q_empty()) return false;
    std::strncpy(out, s_queue[s_q_head], VP_PATH_MAX);
    return true;
}

static void q_pop(void)
{
    if (!q_empty()) s_q_head = (s_q_head + 1) % VP_QUEUE_MAX;
}

/* ---------------- playback engine ---------------- */

/* Advance the file pointer past RIFF header + any non-"data" chunks, leaving
   the read pointer at the start of the PCM samples. Handles WAV variants that
   carry extra chunks (fmt + fact + LIST info before data). */
static bool wav_seek_to_data(FIL *fp)
{
    UINT rd;
    uint8_t riff[12];
    if (f_read(fp, riff, 12, &rd) != FR_OK || rd != 12) return false;
    if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F') return false;
    if (riff[8] != 'W' || riff[9] != 'A' || riff[10] != 'V' || riff[11] != 'E') return false;

    for (int guard = 0; guard < 16; ++guard) {
        uint8_t h[8];
        if (f_read(fp, h, 8, &rd) != FR_OK || rd != 8) return false;
        uint32_t sz = (uint32_t)h[4] | ((uint32_t)h[5] << 8)
                    | ((uint32_t)h[6] << 16) | ((uint32_t)h[7] << 24);
        if (h[0] == 'd' && h[1] == 'a' && h[2] == 't' && h[3] == 'a') return true;
        /* skip this chunk; chunks are word-aligned */
        if (f_lseek(fp, f_tell(fp) + ((sz + 1) & ~1u)) != FR_OK) return false;
    }
    return false;
}

static bool open_next_file(void)
{
    if (s_file_open) { f_close(&s_fil); s_file_open = false; }

    while (!q_empty()) {
        char path[VP_PATH_MAX];
        q_peek(path);
        q_pop();
        if (f_open(&s_fil, path, FA_OPEN_EXISTING | FA_READ) == FR_OK) {
            if (wav_seek_to_data(&s_fil)) {
                s_file_open = true;
                return true;
            }
            std::printf("[voice] bad wav: %s\n", path);
            f_close(&s_fil);
        } else {
            std::printf("[voice] open fail: %s\n", path);
        }
    }
    return false;
}

/* Read 16-bit mono PCM from current file, expand to 16-bit stereo interleaved
   and pack two samples per 32-bit slot (little-endian: low=L, high=R).
   Returns number of 32-bit words written (<= VP_PCM_WORDS).
   When file ends, advances to next segment and continues, so buffers are
   always filled when a clip is available. */
static uint32_t fill_buffer(int32_t *dst)
{
    /* static to avoid a ~4 KB stack allocation per call */
    static int16_t mono[VP_PCM_WORDS];
    uint32_t written = 0;

    while (written < VP_PCM_WORDS) {
        if (!s_file_open) {
            if (!open_next_file()) break;
        }
        UINT need_bytes = (VP_PCM_WORDS - written) * sizeof(int16_t);
        UINT got = 0;
        f_read(&s_fil, mono, need_bytes, &got);
        uint32_t got_samples = got / sizeof(int16_t);

        for (uint32_t i = 0; i < got_samples; ++i) {
            int16_t s = mono[i];
            dst[written + i] = ((uint16_t)s) | ((uint32_t)(uint16_t)s << 16);
        }
        written += got_samples;

        if (got < need_bytes) {      /* file done */
            f_close(&s_fil);
            s_file_open = false;
            /* loop: try next segment to continue filling */
        }
    }

    /* zero-pad tail of buffer if nothing else to play (produces a clean tail) */
    for (uint32_t i = written; i < VP_PCM_WORDS; ++i) dst[i] = 0;
    return written;
}

static void start_dma_if_needed(void)
{
    if (s_playing) return;
    /* Re-arm ch 7 each start: drv_pdma.c's nu_pdma_init (lazy, fires on
       first display PDMA alloc after our own init) does PDMA_Close(all) which
       clears our CHCTL bit. Reassert before every playback. */
    PDMA_Open(VP_PDMA, VP_PDMA_CH_MASK);
    PDMA_SetTransferMode(VP_PDMA, VP_PDMA_CH, PDMA_I2S0_TX, 1, (uint32_t)&s_desc[0]);
    PDMA_CLR_TD_FLAG(VP_PDMA, VP_PDMA_CH_MASK);
    s_playing_idx = 0;
    s_playing = true;
    I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_TX(I2S0);
}

static void stop_dma(void)
{
    I2S_DISABLE_TX(I2S0);
    I2S_DISABLE_TXDMA(I2S0);
    s_playing = false;
    s_buf_full[0] = s_buf_full[1] = 0;
    s_playing_idx = 0;
    /* Re-arm PDMA at desc[0] so the next start_dma begins with buf 0. */
    PDMA_SetTransferMode(VP_PDMA, VP_PDMA_CH, PDMA_I2S0_TX, 1, (uint32_t)&s_desc[0]);
    PDMA_CLR_TD_FLAG(VP_PDMA, VP_PDMA_CH_MASK);
}

/* ---------------- public API ---------------- */

int VoicePlay_Init(void)
{
    if (s_initialised) return 0;

    SYS_UnlockReg();
    vp_clk_init();
    vp_mfp_init();

    I2C_Open(VP_I2C_PORT, 100000);

    /* I2S slave: codec provides BCLK/LRCK, MCU just supplies MCLK */
    I2S_Open(I2S0, I2S_MODE_SLAVE, VP_SAMPLE_RATE, I2S_DATABIT_16, I2S_STEREO, I2S_FORMAT_I2S);

    CLK_SysTickDelay(20000);
    NAU8822_Setup();
    NAU8822_ConfigSampleRate_16k();

    I2S_EnableMCLK(I2S0, 12000000);
    I2S0->CTL0 |= I2S_CTL0_ORDER_Msk;

    vp_pdma_init();

    s_initialised = true;
    std::printf("[voice] init ok\n");
    return 0;
}

void VoicePlay_StopAll(void)
{
    if (!s_initialised) return;
    s_q_head = s_q_tail = 0;
    if (s_file_open) { f_close(&s_fil); s_file_open = false; }
    stop_dma();
}

int VoicePlay_IsBusy(void)
{
    return (!q_empty() || s_file_open || s_playing) ? 1 : 0;
}

void VoicePlay_Speak(int dir, int class_id, int severity)
{
    if (!s_initialised) { std::printf("[voice] Speak: not init\n"); return; }
    if (severity < 1) return;
    if (dir < 0 || dir > 2) return;

    char path[VP_PATH_MAX];
    int pushed = 0;

    std::snprintf(path, VP_PATH_MAX, "0:/voice/dir_%d.wav", dir);
    if (q_push(path)) pushed++;

    if (class_id < 0) {
        std::strcpy(path, "0:/voice/act_unknown.wav");
        if (q_push(path)) pushed++;
    } else {
        std::snprintf(path, VP_PATH_MAX, "0:/voice/cls_%d.wav", class_id);
        if (q_push(path)) pushed++;
        std::strcpy(path, (severity >= 2) ? "0:/voice/act_danger.wav"
                                          : "0:/voice/act_caution.wav");
        if (q_push(path)) pushed++;
    }
    std::printf("[voice] Speak dir=%d cls=%d sev=%d pushed=%d qcnt=%u\n",
                dir, class_id, severity, pushed, (unsigned)q_count());
}

void VoicePlay_TestFile(const char *path)
{
    if (!s_initialised || !path) return;
    q_push(path);
}

void VoicePlay_SetVolume(int db)
{
    if (!s_initialised) return;
    if (db >  6) db =  6;
    if (db < -57) db = -57;
    /* NAU8822 LHP/RHP vol: 0=-57dB, 63=+6dB, 1dB per step. bit8 = update both. */
    uint16_t v = (uint16_t)(db + 57) & 0x3F;
    I2C_WriteNAU8822(52, v);           /* LHPVOL */
    I2C_WriteNAU8822(53, v | 0x100);   /* RHPVOL + HPVU update */
}

/* ===== UART Bluetooth Warning Bridge (to ESP32) ===== */

static bool s_uart_initialised = false;

#define VP_UART_PORT    UART1
#define VP_UART_BAUDRATE 115200
#define UART_START_BYTE 0xAA
#define UART_END_BYTE 0x55

int VoicePlay_UART_Init(void)
{
    if (s_uart_initialised) return 0;
    
    SYS_UnlockReg();
    
    /* Configure UART1: PB3(D1/TX), PB2(D0/RX) - Arduino UNO compatible */
    SYS_ResetModule(SYS_UART1RST);
    CLK_EnableModuleClock(UART1_MODULE);
    CLK_SetModuleClock(UART1_MODULE, CLK_UARTSEL0_UART1SEL_HIRC, CLK_UARTDIV0_UART1DIV(1));
    
    /* Set PB3 as UART1 TX (D1) and PB2 as UART1 RX (D0) */
    SET_UART1_RXD_PB2();
    SET_UART1_TXD_PB3();
    
    /* Open UART */
    UART_Open(VP_UART_PORT, VP_UART_BAUDRATE);
    
    SYS_LockReg();
    
    s_uart_initialised = true;
    std::printf("[UART] Initialized at %d baud on UART1 (D1/PB3 TX, D0/PB2 RX)\n", VP_UART_BAUDRATE);
    return 0;
}

static uint8_t uart_calculate_checksum(uint8_t dir, uint8_t cls, uint8_t sev)
{
    return (dir ^ cls ^ sev);  /* Simple XOR checksum */
}

int VoicePlay_SendWarning(const VoiceWarning_t *warning)
{
    if (!s_uart_initialised || !warning) return -1;
    
    uint8_t packet[6];
    packet[0] = UART_START_BYTE;                                    /* 0xAA */
    packet[1] = warning->direction & 0x03;                          /* 0-2 */
    packet[2] = warning->class_id & 0xFF;                           /* 0-79 or 0xFF */
    packet[3] = warning->severity & 0x03;                           /* 0-2 */
    packet[4] = uart_calculate_checksum(packet[1], packet[2], packet[3]);
    packet[5] = UART_END_BYTE;                                      /* 0x55 */
    
    /* Send 6 bytes */
    for (int i = 0; i < 6; i++) {
        UART_WRITE(VP_UART_PORT, packet[i]);
        while(!UART_IS_TX_EMPTY(VP_UART_PORT));
    }
    
    const char *dir_names[] = {"LEFT", "CENTER", "RIGHT"};
    const char *sev_names[] = {"SAFE", "CAUTION", "DANGER"};
    std::printf("[UART] Sent: dir=%s, cls=%02X, sev=%s\n",
                warning->direction < 3 ? dir_names[warning->direction] : "UNK",
                warning->class_id,
                warning->severity < 3 ? sev_names[warning->severity] : "UNK");
    
    return 0;
}

int VoicePlay_UART_SendBytes(const uint8_t *data, uint32_t length)
{
    if (!s_uart_initialised || !data || length == 0) return -1;

    for (uint32_t i = 0; i < length; i++) {
        UART_WRITE(VP_UART_PORT, data[i]);
        while(!UART_IS_TX_EMPTY(VP_UART_PORT));
    }

    std::printf("[UART] Sent %lu raw bytes\n", (unsigned long)length);
    return 0;
}

void VoicePlay_Pump(void)
{
    if (!s_initialised) return;

    static uint8_t dbg_tick = 0;
    if (!q_empty() || s_file_open || s_playing) {
        if ((++dbg_tick & 0x1F) == 0) {
            std::printf("[pump] play=%d buf=%d%d idx=%d CUR=%08lx CHCTL=%08lx REQ47=%08lx I2Sctl=%08lx DSCT7ctl=%08lx DSCT7next=%08lx\n",
                        s_playing,
                        s_buf_full[0], s_buf_full[1], s_playing_idx,
                        (unsigned long)VP_PDMA->CURSCAT[VP_PDMA_CH],
                        (unsigned long)VP_PDMA->CHCTL,
                        (unsigned long)VP_PDMA->REQSEL4_7,
                        (unsigned long)I2S0->CTL0,
                        (unsigned long)VP_PDMA->DSCT[VP_PDMA_CH].CTL,
                        (unsigned long)VP_PDMA->DSCT[VP_PDMA_CH].NEXT);
        }
    }

    /* We can't trust PDMA TD flag: drv_pdma.c's PDMA0_IRQHandler (line ~1049)
       unconditionally clears ALL channels' TD bits on any PDMA0 IRQ, eating
       ours before we poll it. Instead, detect descriptor flip via CURSCAT. */
    if (s_playing) {
        uint32_t cur = VP_PDMA->CURSCAT[VP_PDMA_CH];
        uint8_t current_idx = (cur == (uint32_t)&s_desc[1]) ? 1
                            : (cur == (uint32_t)&s_desc[0]) ? 0
                            : s_playing_idx;
        if (current_idx != s_playing_idx) {
            s_buf_full[s_playing_idx] = 0;   /* the buf we left is now free */
            s_playing_idx = current_idx;
            PDMA_CLR_TD_FLAG(VP_PDMA, VP_PDMA_CH_MASK);
        }
    }

    /* If nothing more to play, stop. */
    if (q_empty() && !s_file_open && !s_buf_full[0] && !s_buf_full[1]) {
        if (s_playing) stop_dma();
        return;
    }

    /* Refill any empty buffer (check both so we keep the ping-pong fed). */
    for (int i = 0; i < 2; ++i) {
        if (s_buf_full[i]) continue;
        if (q_empty() && !s_file_open) break;

        uint32_t w = fill_buffer(s_pcm[i]);
        if (w == 0) break;

#if (NVT_DCACHE_ON == 1)
        /* Flush our writes so PDMA reads the new data, not stale cache. */
        SCB_CleanDCache_by_Addr((uint32_t *)&s_pcm[i][0], sizeof(s_pcm[0]));
#endif
        s_buf_full[i] = 1;
    }

    /* Start once at least one buffer is primed. Prefer waiting for both so the
       first TD hands off cleanly, but don't stall the very last tail. */
    if (!s_playing && (s_buf_full[0] && s_buf_full[1])) {
        start_dma_if_needed();
    } else if (!s_playing && s_buf_full[0] && q_empty() && !s_file_open) {
        /* only one short clip total */
        start_dma_if_needed();
    }
}
