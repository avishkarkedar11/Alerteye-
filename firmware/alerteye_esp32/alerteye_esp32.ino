/*
  AlertEye AI - ESP32 Hardware Controller
  ==========================================

  Receives status commands over USB Serial from the Python AI app
  (see alerteye/hardware.py):

      NORMAL | CAUTION | DROWSY | CRITICAL | CRITICAL_ESCALATE

  and drives the physical alert hardware (spec section 4):

      Alert state    (NORMAL)   -> Green LED
      Caution state  (CAUTION)  -> Yellow LED
      Drowsiness     (DROWSY)   -> Red LED + buzzer
      Critical       (CRITICAL) -> Red LED flashing + buzzer + vibration
                                    + display "DROWSINESS DETECTED - TAKE A BREAK"

  It also implements an INDEPENDENT safety watchdog (spec section 12):
  if no command arrives from the PC for CONNECTION_TIMEOUT_MS, the ESP32
  assumes the link is dead and goes into a fail-safe "SYSTEM CONNECTION
  LOST" state on its own -- it never silently does nothing.

  A push button lets the driver acknowledge a warning; the firmware
  echoes "ACK" back over serial so the Python side can log driver
  response (spec section 6).

  --------------------------------------------------------------------
  Wiring (adjust pins to your board)
  --------------------------------------------------------------------
    Green LED    -> GPIO 25  (+ 220ohm resistor to GND)
    Yellow LED   -> GPIO 26  (+ 220ohm resistor to GND)
    Red LED      -> GPIO 27  (+ 220ohm resistor to GND)
    Buzzer (+)   -> GPIO 14
    Vibration motor (via NPN transistor / driver) -> GPIO 12
    Push button  -> GPIO 13  (other leg to GND, uses internal pull-up)
    OLED SSD1306 (optional) -> I2C SDA=21, SCL=22

  --------------------------------------------------------------------
  Optional OLED
  --------------------------------------------------------------------
  If you have an SSD1306 128x64 I2C display, install:
    Adafruit_SSD1306 + Adafruit_GFX (Library Manager)
  and set HAS_OLED to 1 below. Without it, status still works fully
  through LEDs/buzzer/vibration and Serial prints.
*/

#include <Arduino.h>

#define HAS_OLED 0

#if HAS_OLED
  #include <Wire.h>
  #include <Adafruit_GFX.h>
  #include <Adafruit_SSD1306.h>
  #define SCREEN_WIDTH 128
  #define SCREEN_HEIGHT 64
  Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
#endif

// ---------------- Pin configuration ----------------
const int PIN_LED_GREEN   = 25;
const int PIN_LED_YELLOW  = 26;
const int PIN_LED_RED     = 27;
const int PIN_BUZZER      = 14;
const int PIN_VIBRATION   = 12;
const int PIN_BUTTON      = 13;

// ---------------- Timing configuration ----------------
const unsigned long CONNECTION_TIMEOUT_MS = 5000;   // spec section 12
const unsigned long CRITICAL_FLASH_MS     = 250;
const unsigned long BUZZER_DROWSY_ON_MS   = 150;
const unsigned long BUZZER_DROWSY_OFF_MS  = 600;
const unsigned long BUTTON_DEBOUNCE_MS    = 250;

// ---------------- State ----------------
enum AlertState { NORMAL, CAUTION, DROWSY, CRITICAL, CRITICAL_ESCALATE, CONNECTION_LOST };
AlertState currentState = NORMAL;

unsigned long lastCommandMillis = 0;
unsigned long lastFlashToggle = 0;
unsigned long lastButtonPress = 0;
bool flashOn = false;
String serialBuffer = "";

void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_VIBRATION, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);

  allOff();

#if HAS_OLED
  if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
  }
#endif

  lastCommandMillis = millis();
  showStatus("BOOT", "Waiting for AI link...");
}

void loop() {
  readSerialCommand();
  checkConnectionWatchdog();
  handleButton();
  driveOutputsForState();
}

// ---------------------------------------------------------------------
// Serial command parsing
// ---------------------------------------------------------------------
void readSerialCommand() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      serialBuffer.trim();
      if (serialBuffer.length() > 0) {
        applyCommand(serialBuffer);
        lastCommandMillis = millis();
      }
      serialBuffer = "";
    } else {
      serialBuffer += c;
    }
  }
}

void applyCommand(const String &cmd) {
  if (cmd == "NORMAL") {
    currentState = NORMAL;
    showStatus("ALERT", "Driver is alert");
  } else if (cmd == "CAUTION") {
    currentState = CAUTION;
    showStatus("CAUTION", "Stay focused");
  } else if (cmd == "DROWSY") {
    currentState = DROWSY;
    showStatus("DROWSY", "Warning issued");
  } else if (cmd == "CRITICAL") {
    currentState = CRITICAL;
    showStatus("CRITICAL", "DROWSINESS DETECTED - TAKE A BREAK");
  } else if (cmd == "CRITICAL_ESCALATE") {
    currentState = CRITICAL_ESCALATE;
    showStatus("CRITICAL!!", "TAKE A BREAK NOW");
  }
  // unknown commands are ignored, link is still considered alive
}

// ---------------------------------------------------------------------
// Independent safety watchdog (spec section 12)
// ---------------------------------------------------------------------
void checkConnectionWatchdog() {
  if (currentState != CONNECTION_LOST &&
      millis() - lastCommandMillis > CONNECTION_TIMEOUT_MS) {
    currentState = CONNECTION_LOST;
    showStatus("NO LINK", "SYSTEM CONNECTION LOST");
  }
}

// ---------------------------------------------------------------------
// Driver acknowledge button (spec section 6 - driver response)
// ---------------------------------------------------------------------
void handleButton() {
  if (digitalRead(PIN_BUTTON) == LOW &&
      millis() - lastButtonPress > BUTTON_DEBOUNCE_MS) {
    lastButtonPress = millis();
    Serial.println("ACK");
  }
}

// ---------------------------------------------------------------------
// Output driving per state
// ---------------------------------------------------------------------
void driveOutputsForState() {
  switch (currentState) {
    case NORMAL:
      setLeds(true, false, false);
      digitalWrite(PIN_BUZZER, LOW);
      digitalWrite(PIN_VIBRATION, LOW);
      break;

    case CAUTION:
      setLeds(false, true, false);
      digitalWrite(PIN_BUZZER, LOW);
      digitalWrite(PIN_VIBRATION, LOW);
      break;

    case DROWSY:
      setLeds(false, false, true);
      digitalWrite(PIN_VIBRATION, LOW);
      pulseBuzzer(BUZZER_DROWSY_ON_MS, BUZZER_DROWSY_OFF_MS);
      break;

    case CRITICAL:
      flashRed(CRITICAL_FLASH_MS);
      digitalWrite(PIN_BUZZER, HIGH);   // continuous
      digitalWrite(PIN_VIBRATION, HIGH);
      break;

    case CRITICAL_ESCALATE:
      flashRed(CRITICAL_FLASH_MS / 2);  // faster flash = more urgent
      digitalWrite(PIN_BUZZER, HIGH);
      digitalWrite(PIN_VIBRATION, HIGH);
      break;

    case CONNECTION_LOST:
      // all LEDs blink together so it's unmistakable from any other state
      flashAll(300);
      pulseBuzzer(80, 1200);            // occasional chirp, not continuous
      digitalWrite(PIN_VIBRATION, LOW);
      break;
  }
}

void setLeds(bool green, bool yellow, bool red) {
  digitalWrite(PIN_LED_GREEN, green ? HIGH : LOW);
  digitalWrite(PIN_LED_YELLOW, yellow ? HIGH : LOW);
  digitalWrite(PIN_LED_RED, red ? HIGH : LOW);
}

void flashRed(unsigned long intervalMs) {
  if (millis() - lastFlashToggle > intervalMs) {
    lastFlashToggle = millis();
    flashOn = !flashOn;
  }
  setLeds(false, false, flashOn);
}

void flashAll(unsigned long intervalMs) {
  if (millis() - lastFlashToggle > intervalMs) {
    lastFlashToggle = millis();
    flashOn = !flashOn;
  }
  setLeds(flashOn, flashOn, flashOn);
}

void pulseBuzzer(unsigned long onMs, unsigned long offMs) {
  unsigned long cyclePos = millis() % (onMs + offMs);
  digitalWrite(PIN_BUZZER, cyclePos < onMs ? HIGH : LOW);
}

// ---------------------------------------------------------------------
// Display / debug output
// ---------------------------------------------------------------------
void showStatus(const char *label, const char *message) {
  Serial.print("[STATE] ");
  Serial.print(label);
  Serial.print(" - ");
  Serial.println(message);

#if HAS_OLED
  display.clearDisplay();
  display.setTextSize(2);
  display.setCursor(0, 0);
  display.println(label);
  display.setTextSize(1);
  display.setCursor(0, 24);
  display.println(message);
  display.display();
#endif
}
