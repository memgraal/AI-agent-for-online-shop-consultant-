"""Interactive terminal client for the SSS chat WebSocket."""

import argparse
import asyncio
import json
import sys
from uuid import uuid4

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError


async def run_session(args: argparse.Namespace) -> bool:
    """Run one chat session. Return True when the user requests a new chat."""
    async with connect(args.url, origin=args.origin, proxy=None) as websocket:
        print("Подключено. Команды: /new — новый диалог, /exit — выход.")
        conversation_id = None
        turn = 0

        while True:
            try:
                message = await asyncio.to_thread(input, "Вы: ")
            except EOFError:
                return False

            command = message.strip().lower()
            if command in {"/exit", "/quit"}:
                return False
            if command == "/new":
                return True
            if not message.strip():
                continue

            turn += 1
            request = {
                "request_id": f"cli-{turn}-{uuid4().hex[:8]}",
                "message": message,
                "stream": args.stream,
                "detail": args.detail,
            }
            if conversation_id is not None:
                request["conversation_id"] = conversation_id

            await websocket.send(json.dumps(request, ensure_ascii=False))
            next_conversation_id = await _print_response(websocket, args.stream)
            if next_conversation_id is not None:
                conversation_id = next_conversation_id


async def _print_response(websocket, streaming: bool) -> str | None:
    """Read a full result or all events for one streamed answer."""
    while True:
        event = json.loads(await websocket.recv())
        event_type = event.get("type")

        if event_type == "result":
            print(f"Ассистент: {event.get('answer', '')}\n")
            return event.get("conversation_id")
        if event_type == "progress":
            print(f"[{event.get('text', '')}]")
        elif event_type == "token":
            sys.stdout.write(event.get("text", ""))
            sys.stdout.flush()
        elif event_type == "done":
            if streaming:
                print("\n")
            return event.get("conversation_id")
        elif event_type == "error":
            print(f"Ошибка: {event.get('message', 'Неизвестная ошибка')}\n")
            return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Интерактивный чат с SSS AI по WebSocket.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/api/chat/ws")
    parser.add_argument("--origin", default="http://localhost:8000")
    parser.add_argument("--detail", choices=("quick", "detailed"), default="quick")
    parser.add_argument("--stream", action="store_true", help="Печатать ответ по мере генерации")
    args = parser.parse_args()

    while True:
        try:
            start_new = await run_session(args)
        except ConnectionClosedError as exc:
            if exc.rcvd is not None and exc.rcvd.code == 1012:
                print("Сервер перезапустился и закрыл соединение. Запустите клиент снова.", file=sys.stderr)
            else:
                print(f"Соединение с сервером закрыто: {exc}", file=sys.stderr)
            return
        except OSError as exc:
            print(f"Не удалось подключиться к {args.url}: {exc}", file=sys.stderr)
            return
        if not start_new:
            return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nЗавершено.")
