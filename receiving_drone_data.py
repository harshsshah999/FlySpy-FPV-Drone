import asyncio
import struct
from bleak import BleakClient, BleakScanner

DEVICE_ADDRESS = "40E6C889-D118-CCD1-0A78-C1DE23559AAD"  # Replace with your actual BLE address
WRITE_CHAR_UUID = "0000abf1-0000-1000-8000-00805f9b34fb"
NOTIFY_CHAR_UUID = "0000abf2-0000-1000-8000-00805f9b34fb"

MSP_GPS = 106
MSP_ATTITUDE = 108

gps_response_event = asyncio.Event()

def create_msp_request(cmd_id: int) -> bytes:
    header = b"$M<"
    size = 0
    checksum = size ^ cmd_id
    return header + bytes([size, cmd_id, checksum])

def parse_msp_gps(payload: bytes):
    if len(payload) != 18:
        print(f"⚠️ GPS payload length mismatch: {len(payload)}")
        return

    lat, lon, alt, speed, course, sats, fix, extra1, extra2 = struct.unpack("<llhhhBBBB", payload)
    
    print(f"\nRaw payload: {payload.hex()}")  # Debug: Print raw payload
    print("\n✅ GPS Data:")
    print(f"  Latitude:   {lat / 1e7} (raw: {lat})")
    print(f"  Longitude:  {lon / 1e7} (raw: {lon})")
    print(f"  Altitude:   {alt} m")
    print(f"  Speed:      {speed} cm/s")
    print(f"  Course:     {course / 10}°")
    print(f"  Satellites: {sats}")
    print(f"  Fix Type:   {'3D' if fix == 2 else 'None'} (raw: {fix})")

def parse_msp_attitude(payload: bytes):
    if len(payload) != 6:
        print(f"⚠️ ATTITUDE payload length mismatch: {len(payload)}")
        return

    roll, pitch, yaw = struct.unpack("<hhh", payload)

    print("\n🧭 Attitude Data:")
    print(f"  Roll:   {roll / 10}°")
    print(f"  Pitch:  {pitch / 10}°")
    print(f"  Yaw:    {yaw / 10}°")

def notification_handler(sender, data):
    print(f"Raw data received: {data.hex()}")  # Debug: Print raw data
    
    if not data.startswith(b"$M>"):
        print(f"Invalid header: {data[:3]}")  # Debug: Print invalid header
        return

    size = data[3]
    cmd = data[4]
    payload = data[5:5 + size]
    checksum = data[5 + size] if len(data) > 5 + size else None

    print(f"Command: {cmd}, Size: {size}, Payload length: {len(payload)}")  # Debug: Print packet info

    calc_checksum = size ^ cmd
    for b in payload:
        calc_checksum ^= b

    if checksum != calc_checksum:
        print(f"Checksum mismatch - Expected: {checksum}, Calculated: {calc_checksum}")  # Debug: Print checksum details
        return

    if cmd == MSP_GPS:
        print("Processing GPS data...")  # Debug: Confirm GPS processing
        parse_msp_gps(payload)
        gps_response_event.set()
    elif cmd == MSP_ATTITUDE:
        print("Processing attitude data...")  # Debug: Confirm attitude processing
        parse_msp_attitude(payload)

async def poll_loop(client):
    try:
        while True:
            gps_response_event.clear()

            await client.write_gatt_char(WRITE_CHAR_UUID, create_msp_request(MSP_GPS))
            await client.write_gatt_char(WRITE_CHAR_UUID, create_msp_request(MSP_ATTITUDE))

            try:
                await asyncio.wait_for(gps_response_event.wait(), timeout=2)
            except asyncio.TimeoutError:
                print("⚠️ No GPS data received (timeout).")

            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("\n🛑 Stopping loop...")

async def discover_devices():
    print("🔍 Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    if not devices:
        print("❌ No BLE devices found!")
        return None
    
    print("\nAvailable devices:")
    for idx, device in enumerate(devices, 1):
        name = device.name or "Unknown"
        print(f"{idx}. {name} ({device.address})")
    
    while True:
        try:
            choice = input("\nSelect device number (or 'q' to quit): ")
            if choice.lower() == 'q':
                return None
            choice = int(choice)
            if 1 <= choice <= len(devices):
                return devices[choice - 1].address
        except ValueError:
            print("Please enter a valid number")
    
    return None

async def main():
    device_address = await discover_devices()
    if not device_address:
        print("❌ No device selected. Exiting...")
        return
    
    print(f"\n🔌 Connecting to {device_address}...")
    async with BleakClient(device_address) as client:
        await client.start_notify(NOTIFY_CHAR_UUID, notification_handler)
        print("📡 Connected and polling every 1 second...")

        poll_task = asyncio.create_task(poll_loop(client))

        try:
            await poll_task
        except KeyboardInterrupt:
            print("\n⏹️ Keyboard interrupt received.")
            poll_task.cancel()
            await asyncio.sleep(1)
        finally:
            await client.stop_notify(NOTIFY_CHAR_UUID)
            print("✅ Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())