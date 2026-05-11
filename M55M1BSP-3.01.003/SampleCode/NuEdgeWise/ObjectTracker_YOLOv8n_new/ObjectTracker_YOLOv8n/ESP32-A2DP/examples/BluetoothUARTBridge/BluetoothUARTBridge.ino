/**************************************************************************//**
 * @file     BluetoothUARTBridge.ino
 * @brief    UART to Bluetooth A2DP Warning Bridge for Visually Impaired System
 *
 * This sketch receives warning signals from M55M1 via UART and converts them
 * into A2DP audio alerts sent to Bluetooth headphones (e.g., WH-1000XM6).
 *
 * UART Protocol:
 *   START_BYTE | DIRECTION | CLASS_ID | SEVERITY | CHECKSUM | END_BYTE
 *      0xAA   |  1 byte   | 1 byte   | 1 byte   | 1 byte   |  0x55
 *
 * Connection:
 *   M55M1 D1/PB1 (TX) -> ESP32 GPIO16 (RX)
 *   M55M1 D0/PB0 (RX) <- ESP32 GPIO17 (TX)
 *   M55M1 GND       -> ESP32 GND
 *
 * Audio Output:
 *   - SAFE:     No sound
 *   - CAUTION:  1000 Hz beep @ 0.5 sec cycle
 *   - DANGER:   1500 Hz beep @ 0.3 sec cycle (fast)
 *   - UNKNOWN:  2000 Hz beep @ 0.4 sec cycle
 *
 * @author  ObjectTracker Vision Team
 * @date    2024-2025
 ******************************************************************************/

#include "BluetoothA2DPSource.h"
#include <HardwareSerial.h>

// ===== UART Configuration =====
#define UART_PORT UART_NUM_1
#define TX_PIN 17                        /* ESP32 GPIO17/TX -> M55M1 D0/PB0 RX */
#define RX_PIN 16                        /* ESP32 GPIO16/RX <- M55M1 D1/PB1 TX */
#define BAUD_RATE 115200

// ===== UART Protocol =====
#define UART_START_BYTE 0xAA
#define UART_END_BYTE 0x55
#define UART_PACKET_SIZE 6
#define UART_TIMEOUT_MS 5000

// ===== Bluetooth Configuration =====
#define TARGET_DEVICE "WH-1000XM6"      /* Sony WH-1000XM6 headphones */
#define SAMPLE_RATE 44100.0f

// ===== Alarm Types =====
enum AlarmType {
    ALARM_SAFE = 0,
    ALARM_CAUTION = 1,
    ALARM_DANGER = 2,
    ALARM_UNKNOWN = 3
};

// ===== Warning Data Structure =====
struct Warning {
    uint8_t direction;  /* 0=LEFT, 1=CENTER, 2=RIGHT */
    uint8_t class_id;   /* 0-79 or 0xFF for unknown */
    uint8_t severity;   /* 0=SAFE, 1=CAUTION, 2=DANGER */
};

// ===== Global Variables =====
BluetoothA2DPSource a2dp_source;
HardwareSerial uart_serial(UART_NUM_1);
static Warning current_warning = {0, 0, 0};
static AlarmType current_alarm = ALARM_SAFE;
static unsigned long last_warning_time = 0;

// ===== A2DP Audio Frame Generation =====
int32_t get_data_frames(Frame *frame, int32_t frame_count) {
    static float t = 0.0;
    
    for (int i = 0; i < frame_count; i++) {
        int16_t sample = 0;
        
        switch(current_alarm) {
            case ALARM_SAFE:
                // No sound
                sample = 0;
                break;
                
            case ALARM_CAUTION:  /* 1000 Hz, 0.5 sec cycle */
                {
                    float phase = fmod(t, 0.5);
                    if (phase < 0.15) {
                        sample = 8000 * sin(2 * PI * 1000 * t);
                    }
                }
                break;
                
            case ALARM_DANGER:   /* 1500 Hz, 0.3 sec cycle (fast) */
                {
                    float phase = fmod(t, 0.3);
                    if (phase < 0.1) {
                        sample = 14000 * sin(2 * PI * 1500 * t);
                    }
                }
                break;
                
            case ALARM_UNKNOWN:  /* 2000 Hz, 0.4 sec cycle */
                {
                    float phase = fmod(t, 0.4);
                    if (phase < 0.2) {
                        sample = 10000 * sin(2 * PI * 2000 * t);
                    }
                }
                break;
        }
        
        frame[i].channel1 = sample;
        frame[i].channel2 = sample;
        t += 1.0 / SAMPLE_RATE;
    }
    
    return frame_count;
}

// ===== UART Packet Parsing =====
uint8_t calculate_checksum(uint8_t dir, uint8_t cls, uint8_t sev) {
    return (dir ^ cls ^ sev);
}

void parse_uart_packet(uint8_t *packet) {
    // Verify packet format
    if (packet[0] != UART_START_BYTE || packet[5] != UART_END_BYTE) {
        Serial.println("[ERROR] Invalid packet format");
        return;
    }
    
    uint8_t dir = packet[1];
    uint8_t cls = packet[2];
    uint8_t sev = packet[3];
    uint8_t checksum = packet[4];
    
    // Verify checksum
    if (checksum != calculate_checksum(dir, cls, sev)) {
        Serial.println("[ERROR] Checksum mismatch");
        return;
    }
    
    // Update warning state
    current_warning.direction = dir;
    current_warning.class_id = cls;
    current_warning.severity = sev;
    last_warning_time = millis();
    
    // Set alarm based on severity and class
    if (sev == 2) {
        current_alarm = ALARM_DANGER;
    } else if (sev == 1) {
        if (cls == 0xFF) {
            current_alarm = ALARM_UNKNOWN;  /* Unknown obstacle */
        } else {
            current_alarm = ALARM_CAUTION;  /* Known object */
        }
    } else {
        current_alarm = ALARM_SAFE;
    }
    
    // Print debug info
    const char *dir_names[] = {"LEFT", "CENTER", "RIGHT"};
    const char *sev_names[] = {"SAFE", "CAUTION", "DANGER"};
    const char *class_name = (cls == 0xFF) ? "UNKNOWN" : "KNOWN";
    
    Serial.printf("[UART RX] dir=%s, cls=%s(%02X), sev=%s\n",
                  dir < 3 ? dir_names[dir] : "INVALID",
                  class_name,
                  cls,
                  sev < 3 ? sev_names[sev] : "INVALID");
}

void handle_uart_receive() {
    static uint8_t packet_buffer[UART_PACKET_SIZE];
    static uint8_t buffer_index = 0;
    
    while (uart_serial.available()) {
        uint8_t byte = uart_serial.read();
        Serial.printf("[RAW] %02X\n", byte);
        
        // Wait for START_BYTE
        if (buffer_index == 0) {
            if (byte == UART_START_BYTE) {
                packet_buffer[0] = byte;
                buffer_index = 1;
            }
            continue;
        }
        
        // Receive subsequent bytes
        packet_buffer[buffer_index] = byte;
        buffer_index++;
        
        // Packet complete
        if (buffer_index == UART_PACKET_SIZE) {
            parse_uart_packet(packet_buffer);
            buffer_index = 0;  /* Reset buffer */
        }
    }
}

// ===== Bluetooth Connection Callback =====
void connection_state_changed(esp_a2d_connection_state_t state, void *ptr) {
    const char *state_str = a2dp_source.to_str(state);
    Serial.printf("[A2DP] Connection state: %s\n", state_str);
    
    if (state == ESP_A2D_CONNECTION_STATE_CONNECTED) {
        Serial.println("[A2DP] ✓ Connected to WH-1000XM6!");
        current_alarm = ALARM_SAFE;  /* Reset alarm on connection */
    } else {
        Serial.println("[A2DP] ✗ Disconnected");
    }
}

// ===== Setup & Loop =====
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n");
    Serial.println("========================================");
    Serial.println(" ESP32 Bluetooth UART Warning Bridge");
    Serial.println("========================================");
    Serial.println("[INFO] Initializing UART...");
    
    // Initialize UART (M55M1 D1/PB1 TX -> ESP32 GPIO16 RX)
    uart_serial.begin(BAUD_RATE, SERIAL_8N1, RX_PIN, TX_PIN);
    Serial.printf("[UART] Initialized at %d baud\n", BAUD_RATE);
    Serial.printf("[UART] RX=GPIO%d <- M55M1 D1/PB1/TX\n", RX_PIN);
    Serial.printf("[UART] TX=GPIO%d -> M55M1 D0/PB0/RX\n", TX_PIN);
    
    // Initialize Bluetooth A2DP
    Serial.println("[INFO] Initializing Bluetooth A2DP...");
    a2dp_source.set_on_connection_state_changed(connection_state_changed);
    a2dp_source.set_data_callback_in_frames(get_data_frames);
    a2dp_source.set_volume(70);  /* 0-127 */
    a2dp_source.start(TARGET_DEVICE);
    
    Serial.printf("[A2DP] Connecting to %s...\n", TARGET_DEVICE);
    Serial.println("========================================\n");
}

void loop() {
    // Continuously receive UART packets
    handle_uart_receive();
    
    // Timeout clear (5 seconds no signal -> SAFE)
    if (millis() - last_warning_time > UART_TIMEOUT_MS && current_alarm != ALARM_SAFE) {
        current_alarm = ALARM_SAFE;
        Serial.println("[TIMEOUT] No warning signal for 5s, returning to SAFE");
    }
    
    delay(10);  /* Prevent watchdog timeout */
}
