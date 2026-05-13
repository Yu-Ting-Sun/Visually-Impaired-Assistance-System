# 完整通訊協議修改清單 (Complete Communication Protocol Changes)

本文檔列出整個ObjectTracker_YOLOv8n專案中所有實施的通訊協議更改。

---

## 1. UART 協議 (UART Communication Protocol)

### 1.1 M55M1 → ESP32 UART 橋接

**文件: [VoicePlayer.hpp](VoicePlayer.hpp)**
- ✅ 新增 `VoiceWarning_t` 結構體 (第 25-29 行)
  ```cpp
  typedef struct {
      uint8_t direction;   /* 0=LEFT, 1=CENTER, 2=RIGHT */
      uint8_t class_id;    /* 0-79 or 0xFF for unknown */
      uint8_t severity;    /* 0=SAFE, 1=CAUTION, 2=DANGER */
  } VoiceWarning_t;
  ```

- ✅ 新增函數聲明 (第 32-35 行)
  ```cpp
  int VoicePlay_UART_Init(void);           /* 初始化 UART0 */
  int VoicePlay_SendWarning(const VoiceWarning_t *warning);  /* 發送警告 */
  ```

---

**文件: [VoicePlayer.cpp](VoicePlayer.cpp)**
- ✅ UART 配置常數 (第 407-411 行)
  ```cpp
  #define VP_UART_PORT    UART0
  #define VP_UART_BAUDRATE 115200
  #define UART_START_BYTE 0xAA
  #define UART_END_BYTE 0x55
  ```

- ✅ 實現 `VoicePlay_UART_Init()` (第 413-433 行)
  - 使用 UART0 (非 UART1)
  - 配置引腳: **PB1 (D1/TX)** 和 **PB0 (D0/RX)**
  - 波特率: **115200 bps**
  - Arduino UNO 相容引腳

- ✅ 實現 `uart_calculate_checksum()` (第 435-437 行)
  ```cpp
  return (dir ^ cls ^ sev);  /* XOR 校驗 */
  ```

- ✅ 實現 `VoicePlay_SendWarning()` (第 439-461 行)
  - 6 位元組封包格式:
    ```
    [0xAA] | [DIR(0-2)] | [CLASS_ID(0-79/0xFF)] | [SEV(0-2)] | [XOR_CHK] | [0x55]
    ```
  - 發送時等待 TX FIFO 清空確保可靠性

---

**文件: [main.cpp](Visually-Impaired-Assistance-System/ObjectTracker_YOLOv8n/main.cpp)**
- ✅ UART 初始化調用 (第 ~750 行)
  ```cpp
  VoicePlay_UART_Init();  /* UART0: PB0(D0/RX), PB1(D1/TX) at 115200 baud */
  ```

- ✅ KNOWN 物體警告發送 (第 ~460-470 行)
  ```cpp
  /* 在 VoicePlay_Speak() 之後 */
  VoiceWarning_t warn;
  warn.direction = (uint8_t)best_direction;
  warn.class_id = (uint8_t)best_class_id;
  warn.severity = (uint8_t)best_severity;
  VoicePlay_SendWarning(&warn);
  ```

---

### 1.2 硬體連接 (Hardware Connections)

**M55M1 → ESP32 UART 線路:**
```
M55M1 D1   (PB1/TX) ──────────────> ESP32 GPIO16 (RX)
M55M1 D0   (PB0/RX) <────────────── ESP32 GPIO17 (TX)
M55M1 GND           ──────────────> ESP32 GND
```

**特性:**
- 3-線 RS-232 UART (TX, RX, GND)
- 無硬體握手 (RTS/CTS)
- 無轉接器所需 (直接引腳對引腳)
- 使用 Arduino UNO 相容引腳 (D0-D13)

---

## 2. A2DP 藍牙協議 (Bluetooth A2DP Protocol)

### 2.1 新增 ESP32 藍牙橋接模組

**文件: [ESP32-A2DP/examples/BluetoothUARTBridge/BluetoothUARTBridge.ino](ESP32-A2DP/examples/BluetoothUARTBridge/BluetoothUARTBridge.ino)** ✨ **新建檔案**

**UART 接收部分:**
- ✅ UART 配置 (第 35-37 行)
  ```cpp
  #define UART_PORT UART_NUM_1
  #define RX_PIN 16          /* ESP32 GPIO16 <- M55M1 D1/TX */
  #define TX_PIN 17          /* ESP32 GPIO17 -> M55M1 D0/RX */
  #define BAUD_RATE 115200
  ```

- ✅ 協議定義 (第 40-44 行)
  - 啟動位元組: `0xAA`
  - 結束位元組: `0x55`
  - 封包大小: 6 位元組
  - 超時: 5000 ms

- ✅ UART 接收實現 (loop 中持續呼叫)
  - 圓形緩衝區接收
  - 6 位元組封包驗證
  - XOR 校驗檢查

**A2DP 音訊部分:**
- ✅ 警告音模式 (enum AlarmType)
  ```cpp
  ALARM_SAFE = 0       /* 無聲 */
  ALARM_CAUTION = 1    /* 1000 Hz @ 0.5 秒週期, 0.15 秒脈衝 */
  ALARM_DANGER = 2     /* 1500 Hz @ 0.3 秒週期, 0.1 秒脈衝 (快速) */
  ALARM_UNKNOWN = 3    /* 2000 Hz @ 0.4 秒週期, 0.2 秒脈衝 */
  ```

- ✅ A2DP 音訊框架生成 (第 ~80-140 行)
  ```cpp
  int32_t get_data_frames(Frame *frame, int32_t frame_count)
  /* 根據 current_alarm 生成正弦波音訊 @ 44100 Hz */
  ```

- ✅ 藍牙連接管理
  ```cpp
  connection_state_changed()  /* A2DP 連接/斷線回調 */
  TARGET_DEVICE "WH-1000XM6" /* 目標裝置名稱 */
  ```

**協議流:**
```
M55M1 UART TX (警告信號)
         ↓
   ESP32 UART RX (接收6位元組)
         ↓
   解析 UART 封包
         ↓
   映射到 AlarmType (0-3)
         ↓
   生成對應頻率 + 週期的正弦波
         ↓
   A2DP 串流到耳機
         ↓
   WH-1000XM6 播放警告音
```

---

## 3. 系統級別通訊協議整合 (System-Level Integration)

### 3.1 多層協議堆棧

```
Layer 4: 應用層 (Application Layer)
├─ ObjectTracker_YOLOv8n
├─ 物體檢測 → 方向判定 → 警告類型 → VoiceWarning_t

Layer 3: 本地音訊協議 (Local Audio Protocol)
├─ I2S0 (M55M1 內部)
├─ NAU8822 音訊編碼器
├─ 3.5mm 耳機孔 (本地播放)

Layer 3: 遠端通訊協議 (Remote Communication Protocol)  ⭐ 新增
├─ UART Serial (M55M1 ↔ ESP32)
├─ 115200 bps, 6 位元組封包

Layer 2: 藍牙協議 (Bluetooth Protocol)  ⭐ 新增
├─ A2DP (Audio/Video Distribution Profile)
├─ ESP32 → WH-1000XM6

Layer 1: 物理層 (Physical Layer)
├─ M55M1: GPIO (D0/D1), I2S0, I2C3
├─ ESP32: GPIO (16/17), Bluetooth Radio
├─ 耳機: 藍牙接收器
```

---

### 3.2 警告狀態機轉換

```
檢測到物體 (KNOWN)
├─ 方向判定: LEFT (0), CENTER (1), RIGHT (2)
├─ 類別ID: 0-79 (YOLO 標籤)
├─ 嚴重性: CAUTION (1) 或 DANGER (2)
└─ 發送 VoiceWarning_t

檢測到未知障礙 (UNKNOWN - 需手動實裝)
├─ 方向判定: LEFT/CENTER/RIGHT
├─ 類別ID: 0xFF (未知)
├─ 嚴重性: 1 (CAUTION)
└─ 發送 VoiceWarning_t

         UART Channel           A2DP Channel
         ────────────          ────────────
VoiceWarning_t          →  ESP32 解析器  →  AlarmType → 正弦波 → 耳機
   (direction)              (UART RX)     (ALARM_*)
   (class_id)
   (severity)
```

---

## 4. 編譯注意事項 (Compilation Notes)

### 4.1 M55M1 專案 (Keil μVision)

**必要修改:**
- [x] VoicePlayer.hpp - 新增 UART 函數聲明
- [x] VoicePlayer.cpp - 實現 UART 初始化 + 發送
- [x] main.cpp - 集成 VoicePlay_UART_Init() 和 VoicePlay_SendWarning()

**預期編譯結果:**
```
Rebuild output
  - 沒有新增錯誤 (UART0 定義在 NuMicro.h)
  - printf 可正常輸出 "[UART] Initialized..." 訊息
```

### 4.2 ESP32 專案 (Arduino IDE)

**必要庫:**
```
ESP32-A2DP by pschatzmann
Arduino Audio Tools by pschatzmann
```

**上傳設置:**
```
Board: ESP32 Dev Module
COM Port: <your serial port>
Upload Speed: 921600 baud
```

**預期行為:**
```
[UART] Initialized at 115200 baud on UART0 (D1/TX, D0/RX)  ← M55M1
[A2DP] ✓ Connected to WH-1000XM6!                           ← ESP32
```

---

## 5. 驗證清單 (Verification Checklist)

- [ ] **硬體連接驗證**
  - [ ] M55M1 D1 (PB1/TX) ← → ESP32 GPIO16 (RX)
  - [ ] M55M1 D0 (PB0/RX) ← → ESP32 GPIO17 (TX)  
  - [ ] M55M1 GND ← → ESP32 GND
  - [ ] 三線連接正確無反向

- [ ] **M55M1 編譯驗證**
  - [ ] 編譯成功 (0 errors, 0 warnings)
  - [ ] VoicePlay_UART_Init() 存在於目的碼
  - [ ] VoicePlay_SendWarning() 存在於目的碼

- [ ] **ESP32 編譯驗證**
  - [ ] BluetoothUARTBridge.ino 編譯成功
  - [ ] ESP32-A2DP 庫正確安裝
  - [ ] 上傳至 ESP32 Dev Board

- [ ] **序列通訊驗證**
  - [ ] 開啟 Arduino Serial Monitor @ 115200 baud
  - [ ] 查看 "[UART] Initialized..." 訊息
  - [ ] 查看 "[A2DP] Connected..." 訊息

- [ ] **端對端測試**
  - [ ] M55M1 檢測到物體 → 播放本地語音警告
  - [ ] M55M1 發送 UART 警告 → ESP32 接收
  - [ ] ESP32 解析 UART 封包 → 更新 current_alarm
  - [ ] A2DP 音訊串流 → 耳機播放對應警告音

---

## 6. 已知限制 (Known Limitations)

1. **UART 超時機制** (5 秒無信號自動清空)
   - 若 UART 線路斷開, ESP32 將在 5 秒後自動恢復至 ALARM_SAFE

2. **未實裝未知障礙物偵測** ⚠️
   - main.cpp 中的未知障礙物部分需額外實裝
   - 建議在 frame_difference 邏輯處新增 VoicePlay_SendWarning() 呼叫

3. **單向通訊** 
   - M55M1 → ESP32: UART 警告信號 ✓
   - ESP32 → M55M1: 無反饋通道 (單向)
   - 若需要狀態回傳, 需擴展 UART 協議

4. **A2DP 音訊品質**
   - 簡單正弦波警告音 (無複雜編碼)
   - 適合警告提醒, 不適合人聲合成

---

## 7. 文件參考 (Documentation References)

- [UART_BLUETOOTH_INTEGRATION.md](UART_BLUETOOTH_INTEGRATION.md) - 詳細的技術文檔
- [BluetoothUARTBridge.ino](ESP32-A2DP/examples/BluetoothUARTBridge/BluetoothUARTBridge.ino) - ESP32 完整程式碼
- [VoicePlayer.hpp](VoicePlayer.hpp) - M55M1 UART 標頭
- [VoicePlayer.cpp](VoicePlayer.cpp) - M55M1 UART 實現 (第 405-461 行)

---

**摘要 (Summary):**
✅ **3 個通訊協議已實裝:**
1. **UART Serial** (M55M1 ↔ ESP32) @ 115200 bps
2. **A2DP Bluetooth** (ESP32 → 耳機) @ 44100 Hz 音訊
3. **I2S Audio** (M55M1 內部, 現有功能保持)

**準備就緒進行硬體組裝與編譯上傳。**

---

*修訂日期: 2025*
*ObjectTracker Vision Team*
