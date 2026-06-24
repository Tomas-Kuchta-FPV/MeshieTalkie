import asyncio
import signal
from meshcore import MeshCore, EventType

# ==========================
# Configuration
# ==========================
SERIAL_PORT = "/dev/ttyACM0"
CHANNEL = 2
POLL_INTERVAL = 0.2
# ==========================

stop_event = asyncio.Event()

async def rx_loop(meshcore):
    while not stop_event.is_set():
        try:
            result = await meshcore.commands.get_msg(timeout=1)
            if result.type == EventType.NO_MORE_MSGS:
                await asyncio.sleep(POLL_INTERVAL)
            elif result.type == EventType.CHANNEL_MSG_RECV:
                msg = result.payload
                print(
                    f"\r[CH{msg.get('channel_idx', '?')}] "
                    f"{msg.get('text', '')}",
                    flush=True
                )
                print("> ", end="", flush=True)
            elif result.type == EventType.CONTACT_MSG_RECV:
                msg = result.payload
                print(
                    f"\r[DM] {msg.get('text', '')}",
                    flush=True
                )
                print("> ", end="", flush=True)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"\rRX error: {e}", flush=True)
            await asyncio.sleep(1)

async def tx_loop(meshcore):
    while not stop_event.is_set():
        try:
            text = await asyncio.to_thread(input, "> ")
            if text.strip().lower() in ("exit", "quit", "/q"):
                print("Disconnecting...")
                stop_event.set()
                break
            if not text.strip():
                continue
            result = await meshcore.commands.send_chan_msg(CHANNEL, text)
            if result.type == EventType.ERROR:
                print(f"Send failed: {result.payload}")
        except EOFError:
            stop_event.set()
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"TX error: {e}")

async def main():
    print(f"Connecting to {SERIAL_PORT}...")
    meshcore = await MeshCore.create_serial(SERIAL_PORT)
    print("Connected")
    print(f"Chatting on channel {CHANNEL}  |  type 'exit' or Ctrl+C to quit\n")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        tasks = [
            asyncio.create_task(rx_loop(meshcore)),
            asyncio.create_task(tx_loop(meshcore)),
        ]
        await stop_event.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        print("\nDisconnected.")
        await meshcore.disconnect()

if __name__ == "__main__":
    asyncio.run(main())