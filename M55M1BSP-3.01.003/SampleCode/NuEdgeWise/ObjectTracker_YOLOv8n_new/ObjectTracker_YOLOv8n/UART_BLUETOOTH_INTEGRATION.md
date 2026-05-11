# M55M1 ↔ ESP32 藍牙警告系統整合說明

## 📋 修改摘要

已完成 **M55M1 UART → ESP32 → 藍牙耳機** 的完整系統整合。以下是所做的所有修改：

---

## 🔧 修改詳情

### **1️⃣ VoicePlayer.hpp - 添加 UART 警告結構體**

**位置：** `ObjectTracker_YOLOv8n/VoicePlayer.hpp` (第 22-34 行)

**新增內容：**
```cpp
typedef struct {
    uint8_t direction;   /* 0=LEFT, 1=CENTER, 2=RIGHT */
    uint8_t class_id;    /* 0-79 or 0xFF for unknown */
    uint8_t severity;    /* 0=SAFE, 1=CAUTION, 2=DANGER */
} VoiceWarning_t;

int VoicePlay_UART_Init(void);
int VoicePlay_SendWarning(const VoiceWarning_t *warning);
```

**功能：** 定義 UART 警告數據結構和函數聲明

---

### **2️⃣ VoicePlayer.cpp - 實現 UART 功能**

**位置：** `ObjectTracker_YOLOv8n/VoicePlayer.cpp` (第 396-464 行)

**新增函數：**

#### `VoicePlay_UART_Init()`
- 配置 UART1：PA8(TX) → ESP32 RX16, PA9(RX) ← ESP32 TX17
- 波特率：115200 baud
- 初始化一次即可

#### `VoicePlay_SendWarning(const VoiceWarning_t *warning)`
- 生成 UART 數據包：
  ```
  0xAA | DIRECTION | CLASS_ID | SEVERITY | CHECKSUM | 0x55
  ```
- XOR 校驗碼防止數據損壞
- 發送到 ESP32

---

### **3️⃣ main.cpp - 集成 UART 警告發送**

#### **修改 #1：主程序初始化（第 738 行）**
```cpp
VoicePlay_UART_Init();  /* Initialize UART1 for Bluetooth bridge */
```

#### **修改 #2：KNOWN 物體警告（第 455-467 行）**
當檢測到已知物體時，同時發送 UART 警告：
```cpp
/* Send warning to ESP32 via UART */
VoiceWarning_t warn;
warn.direction = (uint8_t)best_direction;
warn.class_id = (uint8_t)best_class_id;
warn.severity = (uint8_t)best_severity;
VoicePlay_SendWarning(&warn);
```

#### **修改 #3：UNKNOWN 障礙警告（第 992-1017 行）**
分兩處添加 UART 發送：
- **激活時**（首次檢測到障礙）
- **重複警告時**（每 30 幀重複提醒）

---

### **4️⃣ ESP32 代碼 - 新建 BluetoothUARTBridge.ino**

**位置：** `ESP32-A2DP/examples/BluetoothUARTBridge/BluetoothUARTBridge.ino` (新文件)

**功能模塊：**

#### **UART 接收模塊**
- `handle_uart_receive()` - 接收並緩衝 UART 數據包
- `parse_uart_packet()` - 驗證校驗碼並提取警告信息
- 自動超時清除（5 秒無訊號 → SAFE）

#### **A2DP 音頻生成**
- `get_data_frames()` - 生成實時音頻幀
- 根據警告類型生成不同頻率的蜂鳴音：

| 警報類型 | 頻率 | 週期 | beep 時長 |
|---------|------|------|---------|
| SAFE | 無 | - | - |
| CAUTION | 1000 Hz | 0.5s | 0.15s |
| DANGER | 1500 Hz | 0.3s | 0.1s |
| UNKNOWN | 2000 Hz | 0.4s | 0.2s |

#### **Bluetooth 連接管理**
- `connection_state_changed()` - 監測藍牙連接狀態
- 自動連接 WH-1000XM6 耳機
- 連接成功後重置警報狀態

---

## 🔌 硬件接線圖

```
M55M1 (NuMaker)                ESP32 DevKit
┌─────────────┐               ┌──────────┐
│   PA8 (TX) ─┼───────────────┤ GPIO16   │ ← UART RX
│   PA9 (RX) ─┼───────────────┤ GPIO17   │ → UART TX
│   GND      ─┼───────────────┤ GND      │
└─────────────┘               └──────────┘
                                   │
                              Bluetooth A2DP
                                   │
                            WH-1000XM6 Headphones
```

| 連接 | M55M1 | ESP32 |
|------|--------|---------|
| TX | PA8 | GPIO16 |
| RX | PA9 | GPIO17 |
| GND | GND | GND |

---

## 📊 UART 協議說明

### **數據包格式（6 位元組）**
```
位置  數值        說明
───────────────────────────
0     0xAA       開始標誌
1     direction  方向 (0=LEFT, 1=CENTER, 2=RIGHT)
2     class_id   物體類別 (0-79) 或 0xFF=未知障礙
3     severity   危險等級 (0=SAFE, 1=CAUTION, 2=DANGER)
4     checksum   XOR 校驗碼 (dir ^ cls ^ sev)
5     0x55       結束標誌
```

### **例子**
```
警告：左邊，人（類別 1），危險（2）
原始字節：0xAA 0x00 0x01 0x02 0x03 0x55
          開始  左    人   危險  校驗  結束
```

---

## 🎯 系統工作流程

### **1. M55M1 側**
```
main.cpp 警告邏輯
    ↓
best_severity = 2 (DANGER)
    ↓
VoicePlay_Speak()         ← 本地播報
VoicePlay_SendWarning()   ← 同時通過 UART 發送
    ↓
UART1 TX (PA8) → 115200 baud
```

### **2. 傳輸**
```
M55M1 PA8 (TX)
    │
    └─→ UART 電平轉換器（可選）
         │
         └─→ ESP32 GPIO16 (RX) @ 115200 baud
```

### **3. ESP32 側**
```
UART RX (GPIO16) 接收數據包
    ↓
parse_uart_packet() 驗證
    ↓
設置 current_alarm
    ↓
A2DP 生成音頻幀
    ↓
Bluetooth 發送
```

### **4. 藍牙耳機**
```
接收 A2DP 音頻流
    ↓
播放蜂鳴警報
    ↓
用戶聽到實時警告
```

---

## 💡 關鍵特性

✅ **實時傳輸**：UART 低延遲，毫秒級響應  
✅ **容錯機制**：XOR 校驗碼防止數據損壞  
✅ **超時保護**：5 秒無訊號自動返回 SAFE  
✅ **多警報類型**：KNOWN + UNKNOWN + 低光調適  
✅ **模組化設計**：M55M1 和 ESP32 完全獨立  
✅ **無線輸出**：支援任何 A2DP 藍牙耳機  

---

## 🚀 使用步驟

### **M55M1 編譯和燒錄**
1. 在 Keil 中打開 ObjectTracker 項目
2. 編譯（已包含 UART 初始化和發送功能）
3. 燒錄到 M55M1 板

### **ESP32 編譯和燒錄**
1. 在 Arduino IDE 中打開 `BluetoothUARTBridge.ino`
2. 選擇開發板：**ESP32 Dev Module**
3. 安裝庫：
   - 搜索 `ESP32-A2DP` (pschatzmann) → 安裝
   - 搜索 `Arduino Audio Tools` → 安裝
4. 燒錄到 ESP32

### **首次使用**
1. M55M1 和 ESP32 用 USB 供電
2. 確保 WH-1000XM6 耳機已打開並可被檢測
3. 打開 Arduino IDE 監視器（115200 baud）查看日誌
4. 看到 `[A2DP] ✓ Connected to WH-1000XM6!` 表示成功

---

## 📱 監視器輸出示例

```
========================================
 ESP32 Bluetooth UART Warning Bridge
========================================
[UART] Initialized at 115200 baud
[UART] RX=GPIO16 (M55M1 PA8), TX=GPIO17 (M55M1 PA9)
[INFO] Initializing Bluetooth A2DP...
[A2DP] Connecting to WH-1000XM6...

[A2DP] Connection state: ESP_A2D_CONNECTION_STATE_CONNECTED
[A2DP] ✓ Connected to WH-1000XM6!

[UART RX] dir=LEFT, cls=KNOWN(01), sev=DANGER
[UART RX] dir=CENTER, cls=UNKNOWN(FF), sev=CAUTION
[TIMEOUT] No warning signal for 5s, returning to SAFE
```

---

## 🔧 故障排除

| 問題 | 原因 | 解決方案 |
|------|------|---------|
| ESP32 收不到 UART 數據 | PA8/PA9 接線錯誤 或 波特率不符 | 檢查 PA8→GPIO16, PA9→GPIO17；確認 115200 baud |
| 藍牙無法連接 | 耳機未打開 或 WiFi 干擾 | 重啟耳機；遠離 WiFi 路由器 |
| 警報聲很小 | 音量設置過低 | 修改 ESP32 代碼第 316 行：`a2dp_source.set_volume(70)` (0-127) |
| 卡頓或延遲 | UART 緩衝滿或藍牙頻寬不足 | 檢查 UART 連接；嘗試降低 A2DP 采樣率 |

---

## 📚 相關文件

- **M55M1 側**
  - `VoicePlayer.hpp` - 修改：添加 UART API
  - `VoicePlayer.cpp` - 修改：實現 UART 功能
  - `main.cpp` - 修改：集成警告發送

- **ESP32 側**
  - `BluetoothUARTBridge.ino` - 新建：完整橋接代碼

- **硬件連接**
  - M55M1 UART1：PA8(TX), PA9(RX)
  - ESP32 GPIO：16(RX), 17(TX)
  - 波特率：115200 bps

---

## ✅ 驗證檢查清單

- [ ] M55M1 編譯無錯誤
- [ ] ESP32 編譯無錯誤
- [ ] UART 接線正確（PA8→GPIO16, PA9→GPIO17, GND→GND）
- [ ] 波特率正確（115200 baud）
- [ ] WH-1000XM6 已配對並打開
- [ ] 運行時看到 `[A2DP] ✓ Connected` 消息
- [ ] 當 M55M1 檢測到物體時，ESP32 接收 UART 數據
- [ ] 耳機播放正確的蜂鳴聲

---

## 📞 支持信息

- **UART 校驗**：使用 XOR (dir ^ cls ^ sev)
- **超時時間**：5000 ms（可在 ESP32 代碼修改）
- **藍牙標準**：A2DP（Audio/Video Distribution Profile）
- **兼容設備**：任何支援 A2DP 的藍牙耳機

---

**系統整合完成！** ✨

M55M1 的視覺警告現在可以實時通過藍牙發送到你的耳機了。
