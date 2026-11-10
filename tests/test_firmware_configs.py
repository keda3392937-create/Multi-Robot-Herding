import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_SSID = "YAYA"
EXPECTED_PASSWORD = "kedayaya"


def firmware_files():
    return sorted(
        (PROJECT_DIR / "firmware").glob("car*_sta_repel/car*_sta_repel.ino"),
        key=lambda path: int(re.fullmatch(r"car(\d+)_sta_repel\.ino", path.name).group(1)),
    )


def normalize_firmware(source):
    source = re.sub(r"const uint8_t CAR_ID = \d+;", "const uint8_t CAR_ID = <ID>;", source)
    source = re.sub(
        r'const char \*CAR_NAME = "kedaya\d+";',
        'const char *CAR_NAME = "kedaya<ID>";',
        source,
    )
    return "\n".join(line.rstrip() for line in source.splitlines())


def test_all_50_firmware_files_have_matching_identity_and_new_wifi():
    files = firmware_files()
    assert len(files) == 50

    for expected_id, path in enumerate(files, start=1):
        assert path.name == f"car{expected_id}_sta_repel.ino"
        source = path.read_text(encoding="utf-8")
        assert f'const char *WIFI_SSID = "{EXPECTED_SSID}";' in source
        assert f'const char *WIFI_PASSWORD = "{EXPECTED_PASSWORD}";' in source
        assert f"const uint8_t CAR_ID = {expected_id};" in source
        assert f'const char *CAR_NAME = "kedaya{expected_id}";' in source
        assert path.parent.name == path.stem
        assert "ARDUINO_EVENT_WIFI_STA_DISCONNECTED" in source
        assert "wifi_sta_disconnected.reason" in source
        assert "WiFi.macAddress()" in source
        assert "WiFi.RSSI()" in source
        assert "[TCP] idle timeout" in source


def test_firmware_is_consistent_within_each_hardware_group():
    files = firmware_files()
    legacy_ids = {1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15}
    c5_ids = set(range(1, 51)) - legacy_ids
    for group in (
        [path for path in files if int(re.search(r"car(\d+)_", path.name).group(1)) in legacy_ids],
        [path for path in files if int(re.search(r"car(\d+)_", path.name).group(1)) in c5_ids],
    ):
        normalized_sources = {
            normalize_firmware(path.read_text(encoding="utf-8")) for path in group
        }
        assert len(normalized_sources) == 1


def test_c5_uses_available_header_pins_and_locks_5ghz_before_connecting():
    # Exclude battery sense, PSRAM, USB, UART, RGB and boot-mode pins.
    # GPIO3/25 strap SDIO edges only; these sketches use SPI boot and UART/USB upload.
    allowed_pins = {0, 1, 3, 4, 5, 8, 9, 10, 23, 24, 25}
    c5_ids = {3, 6, 7, *range(16, 51)}
    for path in firmware_files():
        car_id = int(re.search(r"car(\d+)_", path.name).group(1))
        if car_id not in c5_ids:
            continue
        source = path.read_text(encoding="utf-8")
        pins = dict(re.findall(r"const int (A_STBY|B_STBY|M[123]_(?:IN[12]|PWM)) = (\d+);", source))
        assert len(pins) == 11
        assert pins["A_STBY"] == "23"
        assert pins["B_STBY"] == "25"
        assert {int(pin) for pin in pins.values()} == allowed_pins
        motor_pins = [pin for name, pin in pins.items() if name.startswith("M")]
        assert len(set(motor_pins)) == 9
        assert pins["A_STBY"] not in motor_pins
        assert pins["B_STBY"] not in motor_pins
        assert "#if !defined(CONFIG_IDF_TARGET_ESP32C5)" in source
        assert "WiFi.setBandMode(WIFI_BAND_MODE_5G_ONLY)" in source
        assert "WiFi.getBandMode() != WIFI_BAND_MODE_5G_ONLY" in source
        assert "WIFI_BAND_MODE_AUTO" not in source
        assert "WIFI_BAND_MODE_2G_ONLY" not in source
        # Cover both the initial connection and the explicit reconnect path.
        assert source.count("require5GHzOnly();") == 2
        for match in re.finditer(r"WiFi\.begin\(WIFI_SSID, WIFI_PASSWORD\);", source):
            assert "require5GHzOnly();" in source[max(0, match.start() - 220):match.start()]


def test_remaining_12_cars_keep_classic_esp32_wiring_and_wifi():
    legacy_ids = {1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15}
    for path in firmware_files():
        car_id = int(re.search(r"car(\d+)_", path.name).group(1))
        if car_id not in legacy_ids:
            continue
        source = path.read_text(encoding="utf-8")
        assert "CONFIG_IDF_TARGET_ESP32C5" not in source
        assert "WIFI_BAND_MODE_5G_ONLY" not in source
        assert "const int A_STBY = 14;" in source
        assert "const int B_STBY = 13;" in source


def test_c5_keeps_the_original_motor_directions_and_protocol():
    original = firmware_files()[0].read_text(encoding="utf-8")
    c5 = firmware_files()[15].read_text(encoding="utf-8")
    for function in (
        "motorStop", "motorForward", "motorBackward", "stopCar", "moveForward",
        "moveBackward", "moveLeftForward", "moveRightForward", "moveLeftBackward",
        "moveRightBackward", "applyMotionCommand", "handleMessage", "sendDiscoveryReply",
    ):
        pattern = rf"void {function}\([^\n]*\) \{{.*?\n\}}"
        assert re.search(pattern, c5, re.S).group() == re.search(pattern, original, re.S).group()
