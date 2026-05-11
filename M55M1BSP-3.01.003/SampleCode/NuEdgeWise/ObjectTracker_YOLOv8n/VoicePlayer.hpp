/**************************************************************************//**
 * @file     VoicePlayer.hpp
 * @brief    Queue-based WAV voice warning player for EdgeEye guide-vision.
 *
 * Plays short Chinese voice clips (16 kHz / 16-bit mono WAV) stored on SD at
 * 0:/voice/ through I2S0 + on-board audio codec (NAU8822) -> 3.5mm jack.
 * Clips are enqueued as logical segments and played back-to-back.
 *
 *   Init once after SD is mounted:  VoicePlay_Init();
 *   In main loop every iteration:   VoicePlay_Pump();
 *   On warning state change:        VoicePlay_Speak(dir, cls, severity);
 ******************************************************************************/
#ifndef VOICE_PLAYER_HPP
#define VOICE_PLAYER_HPP

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Severity values mirror main.cpp: 0=SAFE, 1=CAUTION, 2=DANGER. */
/* direction: 0=LEFT, 1=CENTER, 2=RIGHT. class_id: 0..79, or -1 for unknown obstacle. */

/* ===== UART Warning Bridge to ESP32 ===== */
typedef struct {
    uint8_t direction;   /* 0=LEFT, 1=CENTER, 2=RIGHT */
    uint8_t class_id;    /* 0-79 or 0xFF for unknown */
    uint8_t severity;    /* 0=SAFE, 1=CAUTION, 2=DANGER */
} VoiceWarning_t;

/* Initialize UART for Bluetooth warning bridge (115200 baud) */
int VoicePlay_UART_Init(void);

/* Send warning to ESP32 via UART */
int VoicePlay_SendWarning(const VoiceWarning_t *warning);

/* Send arbitrary bytes over the same UART link. Useful for bring-up tests. */
int VoicePlay_UART_SendBytes(const uint8_t *data, uint32_t length);

/* Return 0 on success, negative on failure. Call after FATFS mounted. */
int  VoicePlay_Init(void);

/* Must be called frequently from main loop. Drives PDMA refill and queue. */
void VoicePlay_Pump(void);

/* Enqueue a full sentence "[dir] + [class or 有障礙物] + [caution/danger]".
 * severity must be >= 1; severity 0 is ignored.
 * class_id = -1 for the unknown-obstacle path (uses act_unknown.wav). */
void VoicePlay_Speak(int dir, int class_id, int severity);

/* Drop everything queued and stop immediately (e.g. on state clear). */
void VoicePlay_StopAll(void);

/* True if a clip is currently being played or queued. */
int  VoicePlay_IsBusy(void);

/* Debug helper: enqueue a single WAV path directly (for isolation testing). */
void VoicePlay_TestFile(const char *path);

/* Headphone volume. db range: -57..+6 (NAU8822 LHP/RHP volume, 1 dB/step).
 * Typical safe listening: -20 to -10. Call any time after VoicePlay_Init. */
void VoicePlay_SetVolume(int db);

#ifdef __cplusplus
}
#endif

#endif /* VOICE_PLAYER_HPP */
