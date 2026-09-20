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


def test_all_firmware_files_share_the_same_logic():
    normalized_sources = {
        normalize_firmware(path.read_text(encoding="utf-8")) for path in firmware_files()
    }
    assert len(normalized_sources) == 1
